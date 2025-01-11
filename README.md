# EduAI Detector

An AI text detection tool designed for educational use, helping teachers identify potentially AI-generated content in student submissions.

## Features

- AI-generated text detection using statistical analysis
- Simple web interface for text submission
- Detailed analysis metrics and explanations
- RESTful API for integration with other tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tony-42069/eduai-detector.git
cd eduai-detector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

1. Start the server:
```bash
python main.py
```

2. Open your web browser and visit:
- Web Interface: http://127.0.0.1:8000/
- API Documentation: http://127.0.0.1:8000/docs

## Project Structure

```
eduai-detector/
├── src/
│   └── eduai_detector/
│       ├── core/           # Core detection logic
│       ├── interface/      # API and web interface
│       └── utils/          # Utility functions
├── tests/                  # Test files
├── docs/                   # Documentation
└── main.py                # Application entry point
```

## Development

- Built with FastAPI for the backend API
- Uses statistical analysis for AI text detection
- Includes a simple web interface for easy testing

## License

[MIT License](LICENSE)