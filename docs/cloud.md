# Training dispick on a cloud GPU

The training image runs the whole pipeline in one work folder (`scripts/cloud_train.sh`):

1. builds the modal bank (50 000 earth models: about 15 minutes on 16 cores), the validation
   shard (4 096 examples) and the benchmark (2 000 images), unless they are already there;
2. trains, one process per visible GPU, from images synthesized on the CPUs while the GPU
   trains; it resumes from the folder's `last.pt` when started again;
3. exports the best checkpoint to ONNX (`dispick.onnx` and its card `dispick.json`), and
   benchmarks it against the classical picker (`report/report.md`).

What to expect with the default `configs/train_cloud.yaml` (60 000 steps of 48 images): about
3 to 4 hours on one NVIDIA L4 or A10G with 16 vCPUs, so roughly 5 to 7 USD on demand, and
less on spot or preemptible machines. What you get back: `runs/cloud/dispick.onnx`,
`dispick.json`, `report/`, `metrics.jsonl` and TensorBoard logs.

## Choosing a machine

The CPUs matter as much as the GPU: each training image is synthesized on the fly, about
35 ms of one core. Give each GPU at least 12 vCPUs, or the GPU waits.

| Provider | Machine | GPU | vCPU | Notes |
|---|---|---|---|---|
| AWS | `g6.4xlarge` | 1 x L4 24 GB | 16 | the default choice |
| AWS | `g5.4xlarge` | 1 x A10G 24 GB | 16 | as good |
| AWS | `g5.12xlarge` | 4 x A10G | 48 | 4 processes: add `--set data.workers=10` |
| GCP | `g2-standard-16` | 1 x L4 | 16 | the default choice |
| GCP | `g2-standard-48` | 4 x L4 | 48 | 4 processes: add `--set data.workers=10` |

An A100 or H100 with 12 vCPUs is wasted on this: the CPUs cannot feed it.

The host needs an NVIDIA driver of version 560 or newer (the image uses PyTorch built for
CUDA 12.6, whose libraries it carries) and the NVIDIA container toolkit. The providers'
"deep learning" machine images come with both.

## Option A: a GPU virtual machine (EC2 or Compute Engine)

The simplest: a machine you start, run the container on, and stop.

1. Start the machine.
   - AWS: an EC2 `g6.4xlarge` with the *Deep Learning Base OSS Nvidia Driver GPU AMI
     (Ubuntu 22.04)*, and a 100 GB disk.
   - GCP: a Compute Engine `g2-standard-16` with one L4 and a *Deep Learning VM* image
     (a `common-cu12x` family with the NVIDIA driver), and a 100 GB disk.
2. On the machine, get the code and build the image:

   ```sh
   git clone <dispick repository> dispick && cd dispick
   docker build -f docker/Dockerfile -t dispick-train .
   ```

3. Run it, with the work folder on the machine's disk (`--shm-size`: the data workers hand
   batches to the trainer through shared memory):

   ```sh
   mkdir -p ~/dispick-work
   docker run -d --name dispick --gpus all --shm-size=16g \
     -v ~/dispick-work:/work dispick-train
   docker logs -f dispick
   ```

   Stopped or crashed, the same `docker run` (after `docker rm dispick`) resumes it.
4. Fetch the results, for instance
   `scp -r <machine>:dispick-work/runs/cloud/{dispick.onnx,dispick.json,report} .`
   and stop the machine.

## Option B: an AWS SageMaker training job

Managed, and fine on spot capacity: SageMaker syncs `/opt/ml/checkpoints` with S3, so a
reclaimed spot instance resumes where it stopped, and uploads the model at the end.

1. Push the image to ECR (account `123456789012`, region `eu-west-3` here):

   ```sh
   aws ecr create-repository --repository-name dispick-train
   aws ecr get-login-password | docker login --username AWS --password-stdin \
     123456789012.dkr.ecr.eu-west-3.amazonaws.com
   docker tag dispick-train 123456789012.dkr.ecr.eu-west-3.amazonaws.com/dispick-train:latest
   docker push 123456789012.dkr.ecr.eu-west-3.amazonaws.com/dispick-train:latest
   ```

2. Start the job (`job.json` below; the role needs SageMaker's execution permissions and
   access to the bucket):

   ```json
   {
     "TrainingJobName": "dispick-2026-09-25",
     "AlgorithmSpecification": {
       "TrainingImage": "123456789012.dkr.ecr.eu-west-3.amazonaws.com/dispick-train:latest",
       "TrainingInputMode": "File"
     },
     "RoleArn": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
     "OutputDataConfig": {"S3OutputPath": "s3://my-bucket/dispick/output"},
     "CheckpointConfig": {
       "S3Uri": "s3://my-bucket/dispick/work",
       "LocalPath": "/opt/ml/checkpoints"
     },
     "ResourceConfig": {
       "InstanceType": "ml.g5.4xlarge",
       "InstanceCount": 1,
       "VolumeSizeInGB": 100
     },
     "Environment": {"DISPICK_WORK": "/opt/ml/checkpoints"},
     "EnableManagedSpotTraining": true,
     "StoppingCondition": {"MaxRuntimeInSeconds": 43200, "MaxWaitTimeInSeconds": 86400}
   }
   ```

   ```sh
   aws sagemaker create-training-job --cli-input-json file://job.json
   aws sagemaker describe-training-job --training-job-name dispick-2026-09-25
   ```

3. The model is in `s3://my-bucket/dispick/output/dispick-2026-09-25/output/model.tar.gz`
   (`dispick.onnx`, `dispick.json`, `report/`); the whole work folder, data and checkpoints
   included, in `s3://my-bucket/dispick/work`.

## Option C: a GCP Vertex AI custom job

Vertex AI mounts buckets under `/gcs/<bucket>`: the work folder lives there, and the script
copies the data to the machine's disk for training.

1. Push the image to Artifact Registry (project `my-project`, region `europe-west4`):

   ```sh
   gcloud artifacts repositories create dispick --repository-format=docker \
     --location=europe-west4
   gcloud auth configure-docker europe-west4-docker.pkg.dev
   docker tag dispick-train europe-west4-docker.pkg.dev/my-project/dispick/train:latest
   docker push europe-west4-docker.pkg.dev/my-project/dispick/train:latest
   ```

2. Start the job with `job.yaml`:

   ```yaml
   workerPoolSpecs:
     - machineSpec:
         machineType: g2-standard-16
         acceleratorType: NVIDIA_L4
         acceleratorCount: 1
       replicaCount: 1
       diskSpec: {bootDiskType: pd-ssd, bootDiskSizeGb: 100}
       containerSpec:
         imageUri: europe-west4-docker.pkg.dev/my-project/dispick/train:latest
         env:
           - {name: DISPICK_WORK, value: /gcs/my-bucket/dispick}
   ```

   ```sh
   gcloud ai custom-jobs create --region=europe-west4 --display-name=dispick \
     --config=job.yaml
   gcloud ai custom-jobs stream-logs <job id> --region=europe-west4
   ```

3. The results are in `gs://my-bucket/dispick/runs/cloud/`.

## Settings

The script reads environment variables (`-e NAME=value` with `docker run`, `Environment` or
`env` in the jobs):

| Variable | Default | What it sets |
|---|---|---|
| `DISPICK_WORK` | `/work` | The work folder: data, runs |
| `DISPICK_CONFIG` | `configs/train_cloud.yaml` | The training configuration |
| `DISPICK_RUN` | `cloud` | The run's folder under `runs/`: a new name starts afresh |
| `DISPICK_BANK_MODELS` | `50000` | Earth models in the bank |
| `DISPICK_VALIDATION`, `DISPICK_BENCHMARK` | `4096`, `2000` | Sizes of the fixed sets |
| `DISPICK_GPUS` | every visible GPU | Training processes |
| `DISPICK_WORKERS` | every core | Workers for building the data |
| `DISPICK_TRAIN_ARGS` | none | More `dispick train` options, e.g. `--set optim.steps=100000` |
| `DISPICK_LOCAL` | `/tmp/dispick-data` | Local disk the data are copied to for training |

## Training on the local GPU instead

With the GPU free (PACo's vLLM stopped), the same run works without Docker:

```sh
uv sync --extra synth --extra train --extra onnx --extra rocm   # AMD; --extra cuda on NVIDIA
uv run dispick bank build --config configs/bank.yaml --out data/bank.h5 --workers 11
uv run dispick data shard --bank data/bank.h5 --out data/validation.h5 --count 4096 --seed 1001 --workers 11
uv run dispick data benchmark --bank data/bank.h5 --out data/benchmark.h5 --count 2000 --seed 2002 --workers 11
uv run dispick train --config configs/train.yaml
```

## Using the trained model

Copy `dispick.onnx` and `dispick.json` side by side anywhere, and point dispick at them:

```sh
export DISPICK_MODEL=/path/to/dispick.onnx
```

or place them in dispick's package as `src/dispick/models/dispick.onnx` and `dispick.json`,
where `Picker.load()` finds them by default. Inference needs only numpy and onnxruntime:
`pip install "dispick[onnx]"`.
