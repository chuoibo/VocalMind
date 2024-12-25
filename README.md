# 🎵 **VocalMind**: Emphasizes understanding and generating speech with emotion.

**VocalMind** is a Speech AI module application that adapts to user emotions for natural, empathetic communication. It analyzes emotions, adjusts speech tone, and responds with answers that match the user's feelings, creating a human-like conversational experience.

## 💡 Speech Module Pipline

![Speech1](https://github.com/user-attachments/assets/8047cda6-2dec-4561-a6d5-294255ea8141)

This pipeline consists of four key modules. Each module plays a crucial role in the overall system. The pipeline takes input either from a live recording or an audio file stream.
- Speech-to-text
- Emotion analysis 
- Text generation
- Text-to-speech


### Speech to Text
The speech-to-text module utilizes [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-large-960h-lv60) for speech recognition, combined with the spelling correction model [oliverguhr/spelling-correction-english-base](https://huggingface.co/oliverguhr/spelling-correction-english-base) to enhance accuracy.

- For live recording, voice activity detection powered by `WebRTC VAD` runs concurrently with speech recognition and spelling correction.
- For audio input files, the system streams and chunks the file to simulate live recording, enabling simultaneous speech recognition and spelling correction.

This integration ensures real-time speech recognition with accurately corrected outputs.

### Emotion Analysis

The emotion analysis module uses the [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) model to classify user emotions based on the semantic meaning of their words. This model identifies seven emotions: anger, disgust, fear, joy, neutral, sadness, and surprise. However, the system utilizes only six: 
- Anger
- Fear
- Joy
- Neutral
- Sadness
- Surprise. 

This classification allows the system to modify the generated text and speech tone to align with the user’s emotional state.

### Text Generation

For the text generation module, we use [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) to generate user replies. The model is prompted based on the detected emotion to ensure the response aligns with the user’s emotional state. For instance:

- If the emotion is **sad**, the model responds with a **sympathetic tone**, addressing the user's concerns with care and understanding.
- If the emotion is **joy**, the model replies in a **cheerful and uplifting manner**, amplifying the positivity in the conversation.
- If the emotion is **surprise**, the model adapts to a **curious and engaging tone**, reflecting excitement or amazement while exploring the user’s input further.
- If the emotion is **anger**, the model takes a **calm and composed approach**, aiming to defuse the tension and provide thoughtful, neutral responses to address the user's concerns.
- If the emotion is **fear**, the model adopts a **reassuring and supportive tone**, offering encouragement and guidance to help ease the user’s apprehension.

By tailoring prompts for each emotion, the model produces appropriate and thoughtful responses, creating a more personalized and engaging interaction.

### Text to Speech


For the text-to-speech module, instead of directly generating speech with emotion, I propose using voice cloning for emotional tones. This approach provides fast inference and accurate imitation of emotional tones such as happy, sympathetic, sad, or surprised. The model used for this module is [xTTSv2](https://huggingface.co/coqui/XTTS-v2) by coqui.

In addition to voice cloning, reference voices are mapped to specific emotional scenarios to ensure the response tone aligns with the user's emotion:

- **Sad**, **Fear**, and **Angry**: Use a sympathetic tone to comfort and calm the user.
- **Happy**: Respond with a joyful tone to maintain an upbeat and engaging conversation.
- **Surprise**: Use a curious tone to match and amplify the user’s sense of wonder or excitement.

This mapping ensures the synthesized speech aligns with the user's emotional state, enhancing the quality and emotional depth of interactions.


## 🎼 Demonstration

## 🎷 System Architecture

This system is designed using a microservices architecture, where the Speech Service and the Database Service communicate with each other through APIs (implemented with FastAPI). The system accepts two types of input: live audio recordings or an audio file provided via its file path.

![architecture](https://github.com/user-attachments/assets/07fe40b7-c649-4662-bc31-310b9174dc66)

### Input (Client to FastAPI):
- **Live Recording**: The client can record audio live, which is sent directly to **FastAPI**.
- **Audio File Path**: The client can provide an audio file path. The **FastAPI** service uploads this file to an **S3 bucket** (`1.1* Uploading Audio File Path`).

### 1. Task Enqueueing (FastAPI to RabbitMQ):
- Once the input (live or file path) is received, **FastAPI** enqueues the task into RabbitMQ (`2.1 Enqueue Task`).
- **RabbitMQ** returns a task ID or relevant task information to **FastAPI** (`2.3 Return task IO`).

### 2. File Retrieval and Task Execution:
- For file-based tasks, the Celery Worker retrieves the audio file from the **S3 bucket** (`3.1* Query File from Bucket`).
- The task is then started by **RabbitMQ**, and it is picked up by a **Celery Worker**, specifically the **Speech Worker** (`3. START TASK`).

### 3. Processing and Status Storage:
- The **Speech Worker** processes the task using the **Speech Module** for operations
- Task status and results are stored in Redis for quick access (`4. Store Task Status`).

### 4. Task Status Querying:
- **RabbitMQ** interacts with **Redis** to fetch the task status when queried by **FastAPI** (`5. Query Task Status`).

### 5. Metadata Management (FastAPI to Database):
- Before processing begins, **FastAPI** updates the **Database** with initial metadata, such as task ID, input type, and timestamps (`2.2 Update Initial Metadata`).
- During or after processing, **FastAPI** retrieves task results or status from **Redis** and updates the **Database** with the final metadata and results (`7. Update Metadata`).

### Output:
- **FastAPI** sends the task results or status back to the Client as output (`8. Output`).


## 📦 Installation

To install the necessary dependencies for each service, please follow the specific criteria and execute the corresponding commands below.

```python
pip install -r api/requirements.txt
pip install -r services/speech/requirements.txt
pip install -r services/database/requirements.txt
```

Ensure that [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html), [Docker Engine](https://docs.docker.com/engine/install/ubuntu/), and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) are installed, as these are the highest prerequisites.

## 🤖 Inference

Certain prerequisites must be prepared before running inference.

### 🛠️ Common Directory

Follow the descriptions of each module in the system pipeline to download the required models and organize them according to the specified structure below. For modules supporting `ONNX`, convert the models to `ONNX` format and place them in the correct directories as indicated.

```
├── emotion
│   ├── config.json
│   ├── merges.txt
│   ├── model.onnx
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── speech2txt
│   └── models--facebook--wav2vec2-base-960h
├── text_gen
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── text_processing
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
└── txt2speech
    ├── models
    │   ├── config.json
    │   ├── hash.md5
    │   ├── model.pth
    │   ├── speakers_xtts.pth
    │   └── vocab.json
    └── ref
        ├── anger.wav
        ├── happy.wav
        ├── neutral.wav
        ├── sad.wav
        ├── surprise.wav
        └── sympathy.wav
```


For the reference emotion audio file paths, please organize, store, and rename the files according to the proposed structure. The dataset containing the emotion reference file paths can be found [HERE](https://github.com/HLTSingapore/Emotional-Speech-Data).

### 💻 Environments File

Create a `.env` file using the sample template below and place it in the appropriate directories for each `api`, `database`, and `speech` service:

```

#General
GPU_COUNT=1
DATABASE_API_URL=http://database:8000

#AWS
AWS_ACCESS_KEY_ID=YOUR_AWS_KEY_ID
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_KEY
AWS_REGION=YOUR_AWS_REGION

# RabbitMQ
RMQ_USER=USER
RMQ_PWD=PASSWORD
MQ_URL=amqp://${RMQ_USER}:${RMQ_PWD}@rabbitmq:5672/

# Redis
REDIS_PWD=PASSWORD
REDIS_URL=redis://:${REDIS_PWD}@redis:6379/0

#Database
DATABASE_USER=USER
DATABASE_PASSWORD=PASSWORD
DATABASE_HOSTNAME=mongodb
DATABASE_PORT=27017
DATABASE_URL=mongodb://${DATABASE_USER}:${DATABASE_PASSWORD}@${DATABASE_HOSTNAME}:${DATABASE_PORT}

```

If you don’t yet have an S3 bucket on AWS for storage, please register for one and obtain the necessary access keys for the AWS bucket.

### ▶️ QUICK START

Here's a basic command line to run:

```python
### For running docker compose
bash script/dev-full.sh

### For rebuilding then running docker compose
bash script/dev-full.sh rebuild

```

You can monitor the speech tasks using Flower by opening `localhost:5555`. To interact with FastAPI for sending and receiving tasks, open `localhost:8081/docs`.



