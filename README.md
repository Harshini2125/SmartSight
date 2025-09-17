# SmartSight
SmartSight: A multilingual, AI-powered navigation and safety assistant that provides real-time obstacle detection and context-aware voice guidance for visually impaired users.
Core Components

VoiceAssistant Class

Speech recognition and synthesis
Language management
Audio processing


LLMAssistant Class

Multiple AI provider integration
Response generation and translation
Fallback handling


GPSNavigator Class

Real-time location services
Route calculation and optimization
Geocoding and address resolution


IntelligentAssistant Class

Command processing and routing
Emergency detection and handling
Mode management



Data Flow
User Voice Input → Speech Recognition → Intent Analysis → 
AI Processing → Response Generation → Translation → 
Text-to-Speech → Audio Output
📊 Performance Monitoring
The system automatically logs performance metrics to performance_log.csv:

Response Times: Voice recognition, AI processing, navigation
Success Rates: Feature usage and error tracking
User Interactions: Command patterns and language preferences
System Health: API availability and service status

🚨 Emergency Features
India-Specific Emergency Contacts

Police: 100
Fire Department: 101
Ambulance: 102
Universal Emergency: 112

Emergency Capabilities

Voice-activated emergency reporting
Automatic location sharing
Quick access emergency buttons
Multi-language emergency communication

🌍 Language Support
LanguageCodeVoice InputVoice OutputText TranslationEnglishen-US✅✅✅Hindihi-IN✅✅✅Telugute-IN✅✅✅Spanishes-ES✅✅✅Frenchfr-FR✅✅✅Germande-DE✅✅✅Italianit-IT✅✅✅Portuguesept-PT✅✅✅Russianru-RU✅✅✅Chinesezh-CN✅✅✅Japaneseja-JP✅✅✅Koreanko-KR✅✅✅Arabicar-SA✅✅✅
🛠️ Configuration
Audio Settings
python# Speech recognition sensitivity
recognizer.energy_threshold = 300
recognizer.pause_threshold = 0.8

# Text-to-speech settings  
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
Navigation Settings
python# Location services timeout
timeout=15

# Route calculation preferences
profile='driving'  # or 'walking'
🐛 Troubleshooting
Common Issues
Microphone not detected:
bash# Test microphone access
python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"
Audio playback issues:
bash# Install additional audio dependencies
pip install playsound pygame pyaudio
API connection errors:

Verify API keys are correctly configured
Check internet connectivity
Monitor API rate limits

Navigation not working:

Ensure location services are enabled
Check OSRM server availability
Verify geocoding service access

Debug Mode
Enable detailed logging by setting:
pythonlogging.basicConfig(level=logging.DEBUG)
🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Development Guidelines

Follow PEP 8 style guidelines
Add comprehensive error handling
Include performance logging for new features
Test multilingual functionality
Document API integrations

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

OpenStreetMap: Map data and routing services
Google Speech API: Speech recognition services
gTTS: Text-to-speech synthesis
Streamlit: Web application framework
Folium: Interactive mapping capabilities

📞 Support
For support and questions:

Create an issue on GitHub
Check the troubleshooting section
Review performance logs for debugging

🚀 Future Enhancements

 Offline voice recognition capability
 Custom wake word detection
 Integration with smart home devices
 Advanced traffic-aware routing
 Multi-user conversation support
 Voice biometric authentication
 Integration with calendar and reminders
 Real-time language translation during calls


⚠️ Important Note: This application is designed for assistance and educational purposes. For actual emergencies, always contact emergency services directly.
