# Chatbot Deployment

Welcome to the Chatbot Deployment repository. This application uses a Bert Model to generate chatbot responses.

## Live Chatbot Interface

You can interact with the chatbot by visiting the following link: [Chatbot Interface](https://bertbasechatbot.ngrok.io ). The chatbot's opening page is located in the bottom right corner of the page.

## Dataset and Model Customization

By default, the application uses a limited dataset. Users have the option to choose their own dataset or customize the model for content generation.

## Prerequisites

- Python 3.10

## Installation and Training

Follow the steps below to set up your environment and train the Bert Model:

```bash
# Navigate to the chatbot deployment directory
cd chatbot-deployment

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required dependencies
pip install -r requirements.txt

# Train the Bert Model
python train.py
```
## Running the Chatbot
After training the model, you can start the chat interface using the following command:

```bash
python chat.py
```

## Troubleshooting
If you encounter any issues with the accelerate package, you can force a reinstallation with:
```bash
pip install --upgrade --force-reinstall accelerate
```

If you experience any other issues or have questions, feel free to open an issue in this repository.

