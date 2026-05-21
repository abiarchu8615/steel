# AI Digital Twin for Steel Manufacturing Predictive Maintenance

This Streamlit prototype uses the AI4I 2020 Predictive Maintenance dataset to monitor steel manufacturing machine-health signals and generate autonomous maintenance tickets with Telegram/email alerts.

## Features

- Machine health dashboard
- Machine failure prediction demo
- Failure trend prediction
- Severity score and risk probability
- Autonomous maintenance ticket generation
- Telegram and email alert buttons
- AI chatbot-style maintenance assistant

## Dataset

Place `ai4i2020.csv` inside the `data/` folder.

## Setup

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Streamlit Secrets

Create `.streamlit/secrets.toml`:

```toml
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_telegram_chat_id"

EMAIL_SENDER = "your_sender_email@gmail.com"
EMAIL_PASSWORD = "your_gmail_app_password"
EMAIL_RECEIVER = "receiver_email@gmail.com"
```

Do not commit your real secrets to GitHub.
