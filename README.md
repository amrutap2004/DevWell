# DevWellApp - Developer Wellness Monitor

A comprehensive desktop application that monitors developer health and wellness during work sessions using computer vision and AI.

## Features

### 🔍 **Eye Health Monitoring**
- Real-time eye strain detection using MediaPipe
- Blink rate analysis (alerts if < 17 blinks/minute)
- Eye aspect ratio tracking
- Tiredness detection for extended eye closure

### 🏃 **Posture Analysis**
- Body posture scoring using pose estimation
- Bad posture alerts after 2 minutes of poor alignment
- Real-time posture feedback

### 📊 **Activity Tracking**
- Keyboard and mouse activity monitoring
- Session duration tracking
- Break reminders and suggestions
- Multi-user support with face recognition

### 📈 **Health Analytics**
- Generate daily/weekly health reports
- Export data to CSV format
- Visual charts and graphs using matplotlib
- SQLite database for data persistence

### 🔔 **Smart Notifications**
- Voice alerts using text-to-speech
- System notifications
- Customizable alert thresholds
- Health tips and break suggestions

## Quick Start

### Option 1: Run Setup Script (Recommended)
```bash
python setup.py
```

### Option 2: Manual Installation
```bash
pip install -r requirements.txt
python DevWellApp.py
```

## System Requirements
- Windows 10/11
- Python 3.8+
- Webcam
- 4GB RAM (8GB recommended)

## Usage

1. **Launch Application**: Run `DevWellApp.py` or use desktop shortcut
2. **Start Monitoring**: Click "Start Tracking" button
3. **View Reports**: Use "Generate Report" for health analytics
4. **Adjust Settings**: Configure thresholds and preferences
5. **Take Breaks**: Follow break suggestions for optimal health

## Application Structure

```
DevWellApp/
├── DevWellApp.py          # Main application (1500+ lines)
├── requirements.txt       # Dependencies
├── setup.py              # Automated setup
├── README.md             # This file
├── DEPLOYMENT_GUIDE.md   # Detailed deployment instructions
└── devwell.db            # SQLite database (auto-created)
```

## Key Components

- **Main Window**: PyQt5-based GUI with real-time monitoring display
- **Computer Vision**: OpenCV + MediaPipe for face/pose detection
- **Database**: SQLite3 for storing health metrics and settings
- **Audio System**: pyttsx3 for voice notifications
- **Input Monitoring**: pynput for keyboard/mouse activity tracking

## Health Metrics Tracked

- Eye blink frequency
- Eye aspect ratio (EAR)
- Posture alignment score
- Keyboard activity levels
- Mouse movement patterns
- Session duration
- Break frequency
- Alert counts by type

## Customization

Access settings through the application menu to adjust:
- Alert thresholds
- Monitoring intervals
- Voice notification preferences
- Report generation periods
- Break reminder frequency

## Data Privacy

- All data stored locally in SQLite database
- No network connections required
- No data transmitted externally
- Complete user privacy maintained

## Troubleshooting

**Camera Issues**: Ensure webcam is connected and not used by other applications
**Performance**: Close unnecessary applications for better face detection
**Permissions**: Run as administrator if experiencing access issues

## Contributing

This is a desktop health monitoring application designed for developer wellness. For modifications or enhancements, refer to the main `DevWellApp.py` file structure.

## License

Personal/Educational use. Please respect privacy and health monitoring ethics when using this application.
