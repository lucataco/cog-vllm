# Cog-vLLM: Run vLLM on Replicate

[Cog](https://github.com/replicate/cog) 
is an open-source tool that lets you package machine learning models
in a standard, production-ready container. 
vLLM is a fast and easy-to-use library for LLM inference and serving.

You can deploy your packaged model to your own infrastructure, 
or to [Replicate].

## Highlights

* 🚀 **Run vLLM in the cloud with an API**.
  Deploy any [vLLM-supported language model] at scale on Replicate.

* 🏭 **Support multiple concurrent requests**.
  Continuous batching works out of the box.

* 🐢 **Open Source, all the way down**.
  Look inside, take it apart, make it do exactly what you need.

## Getting Started

If you're on a machine or VM with a GPU,
you can run vLLM models locally using this cog-vllm wrapper.

### Prerequisites

You'll need:
- A URL to a `.tar` archive containing your model weights, or a local path to model files
- Cog installed ([v0.10.0-alpha11](https://github.com/replicate/cog/releases/tag/v0.10.0-alpha11) or later)
- A GPU-enabled machine for optimal performance

### Installation

Start by [installing or upgrading Cog](https://cog.run/#install):

```console
$ sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/download/v0.10.0-alpha11/cog_$(uname -s)_$(uname -m)"
$ sudo chmod +x /usr/local/bin/cog
```

Then clone this repository:

```console
$ git clone https://github.com/replicate/cog-vllm
$ cd cog-vllm
```

Set the `COG_WEIGHTS` environment variable with your model weights URL or local path: 

```console
$ export COG_WEIGHTS="https://your-weights-url.com/model.tar"
# or for local development:
$ export COG_WEIGHTS="/path/to/local/model/directory"
```

### Running Predictions

Make your first prediction against the model locally:

```console
$ cog predict -e "COG_WEIGHTS=$COG_WEIGHTS" \ 
              -i prompt="Hello!"
```

The first time you run this command with a URL,
Cog downloads the model weights and saves them to the local directory.

To make multiple predictions,
start up the HTTP server and send it `POST /predictions` requests:

```console
# Start the HTTP server
$ cog run -p 5000 -e "COG_WEIGHTS=$COG_WEIGHTS" python -m cog.server.http

# In a different terminal session, send requests to the server
$ curl http://localhost:5000/predictions -X POST \
    -H 'Content-Type: application/json' \
    -d '{"input": {"prompt": "Hello!"}}'
```

## Deploying to Replicate

When you're ready to deploy your model to Replicate,
you can push your changes:

Grab your token from [replicate.com/account](https://replicate.com/account) 
and set it as an environment variable:

```shell
export REPLICATE_API_TOKEN=<your token>
```

```console
$ echo $REPLICATE_API_TOKEN | cog login --token-stdin
$ cog push r8.im/<your-username>/<your-model-name>
--> ...
--> Pushing image 'r8.im/...'
```

### Using Your Model on Replicate

After you push your model, you can run it via the Replicate API.

Install the [Replicate Python SDK][replicate-python]:

```console
$ pip install replicate
```

Create a prediction and stream its output:

```python
import replicate

model = replicate.models.get("<your-username>/<your-model-name>")
prediction = replicate.predictions.create(
    version=model.latest_version,
    input={"prompt": "Hello"},
    stream=True
)

for event in prediction.stream():
    print(str(event), end="")
```

> **Note**: When deploying to Replicate, you'll need to ensure your model has access to the weights.
> You can either bake the weights into your model image or provide the `COG_WEIGHTS` URL at runtime.

[Replicate]: https://replicate.com
[vLLM-supported language model]: https://docs.vllm.ai/en/latest/models/supported_models.html
[replicate-python]: https://github.com/replicate/replicate-python
