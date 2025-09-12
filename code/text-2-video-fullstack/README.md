# Text-to-Video Fullstack App

A fullstack application that generates videos from text prompts using Google's Veo 3.0 model. The app features a modern Gradio web interface for easy interaction.

[![AI Video Generator](https://img.shields.io/badge/AI%20Video%20Generator-FF0000?logo=youtube&logoColor=white&style=for-the-badge)](https://www.youtube.com/watch?v=173br2ADiuw)

## Features

- **Text-to-Video Generation**: Convert text prompts into high-quality videos using Google's Veo 3.0 model
- **Web Interface**: Clean and intuitive Gradio-based UI for easy interaction
- **Docker Support**: Containerized deployment for easy setup and scaling
- **Authentication**: Built-in authentication for secure access
- **Real-time Processing**: Live status updates during video generation
- **Deployment**: Applicaton can be easily deployed to any Cloud Provider. Watch video to deploy on Hugging Face Spaces.

## Prerequisites

- Python 3.11+
- Google API Key with access to Veo 3.0 model
- Docker (optional, for containerized deployment)

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd text-2-video-fullstack
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your Google API key
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

4. Run the application:
```bash
python ui.py
```

The app will be available at `http://localhost:7860`

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t text-2-video-app .
```

2. Run the container:
```bash
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_api_key_here text-2-video-app
```

## Usage

1. Open the web interface in your browser
2. Enter a descriptive text prompt for the video you want to generate
3. Click "Generate Video" and wait for the process to complete
4. View and download your generated video

### Example Prompts

- "A high-end fashion ad showing a confident male model in a new dark blue suit sprinting down a professional runway"
- "A serene sunset over a calm ocean with gentle waves"
- "A futuristic cityscape with flying cars and neon lights"

## API Configuration

The app uses Google's Veo 3.0 model with the following default settings:
- **Aspect Ratio**: 16:9
- **Negative Prompt**: "cartoon, drawing, low quality"
- **Model**: veo-3.0-generate-001

## Authentication

The app includes basic authentication:
- **Username**: devmode
- **Password**: testdeployment8721

*Note: Change these credentials in production environments*

## Project Structure

```
text-2-video-fullstack/
├── ui.py              # Gradio web interface
├── veo_vid.py         # Video generation logic
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container configuration
├── .env.example       # Environment variables template
└── README.md          # This file
```

## Dependencies

- **google-genai**: Google's Generative AI client library
- **gradio**: Web UI framework for machine learning apps
- **python-dotenv**: Environment variable management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is part of the awesome-genai-apps collection.

## Support

For issues and questions, please open an issue in the repository.
