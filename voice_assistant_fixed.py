import os
import time
import json
import math
import requests
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import threading
import tempfile
import random
import io

# Core Libraries
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import pygame

# Web Framework
import streamlit as st
from streamlit_folium import st_folium

# GPS and Location
from geopy.geocoders import Nominatim
import folium

######################################################################
# Enhanced LLM Integration with Better Multilingual Support
######################################################################

class LLMAssistant:
    """Enhanced AI assistant with multiple LLM providers and better multilingual support"""
    
    def __init__(self):
        # Multiple LLM endpoints for better reliability
        self.llm_endpoints = [
            {
                'name': 'Groq',
                'url': 'https://api.groq.com/openai/v1/chat/completions',
                'headers': {
                    'Authorization': 'Bearer gsk_demo_key',  # Replace with actual key
                    'Content-Type': 'application/json'
                },
                'model': 'llama3-8b-8192'
            },
            {
                'name': 'OpenRouter',
                'url': 'https://openrouter.ai/api/v1/chat/completions',
                'headers': {
                    'Authorization': 'Bearer sk-or-demo',  # Replace with actual key
                    'Content-Type': 'application/json'
                },
                'model': 'microsoft/wizardlm-2-8x22b'
            },
            {
                'name': 'Together',
                'url': 'https://api.together.xyz/v1/chat/completions',
                'headers': {
                    'Authorization': 'Bearer demo_key',  # Replace with actual key
                    'Content-Type': 'application/json'
                },
                'model': 'meta-llama/Llama-2-7b-chat-hf'
            },
            # Add OpenAI as backup
            {
                'name': 'OpenAI',
                'url': 'https://api.openai.com/v1/chat/completions',
                'headers': {
                    'Authorization': 'Bearer sk-demo_key',  # Replace with actual key
                    'Content-Type': 'application/json'
                },
                'model': 'gpt-3.5-turbo'
            }
        ]
    
    def get_llm_response(self, question: str, target_language: str = 'en') -> str:
        """Get response from LLM providers with enhanced multilingual support"""
        
        # Enhanced language instruction mapping
        language_instructions = {
            'en': 'English',
            'hi': 'Hindi (हिंदी) - Write in Devanagari script',
            'te': 'Telugu (తెలుగు) - Write in Telugu script',
            'es': 'Spanish (Español) - Write in Spanish',
            'fr': 'French (Français) - Write in French',
            'de': 'German (Deutsch) - Write in German',
            'it': 'Italian (Italiano) - Write in Italian',
            'pt': 'Portuguese (Português) - Write in Portuguese',
            'ru': 'Russian (Русский) - Write in Cyrillic script',
            'zh': 'Chinese (中文) - Write in Chinese characters',
            'ja': 'Japanese (日本語) - Write in Japanese script',
            'ko': 'Korean (한국어) - Write in Korean script',
            'ar': 'Arabic (العربية) - Write in Arabic script'
        }
        
        lang_instruction = language_instructions.get(target_language, 'English')
        
        # Enhanced system prompt for better responses
        system_prompt = f"""You are a helpful voice assistant like Google Assistant. 
CRITICAL: You must respond ONLY in {lang_instruction} language. Do not use English if the target language is not English.

Guidelines:
- Be conversational and natural like a real voice assistant
- Keep responses concise (2-3 sentences maximum)
- For recipes (like biryani): Give brief overview with key steps
- For general knowledge: Provide accurate, helpful information
- For current time/date: Use the information provided or mention checking device
- For weather: Suggest checking weather apps briefly
- For navigation: Acknowledge and suggest using navigation mode
- Be friendly and helpful like Google Assistant

IMPORTANT: Write your entire response in {lang_instruction}. If you don't know the target language well, provide a simple response in that language."""

        user_prompt = f"Question: {question}\n\nRespond naturally in {lang_instruction} like a voice assistant would."
        
        # Try each LLM endpoint with shorter timeout
        for endpoint in self.llm_endpoints:
            try:
                payload = {
                    "model": endpoint['model'],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 150,  # Reduced for faster responses
                    "temperature": 0.7,
                    "stream": False
                }
                
                response = requests.post(
                    endpoint['url'],
                    headers=endpoint['headers'],
                    json=payload,
                    timeout=8  # Reduced timeout for speed
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'choices' in data and len(data['choices']) > 0:
                        answer = data['choices'][0]['message']['content'].strip()
                        if answer:
                            return answer
                            
            except Exception as e:
                continue
        
        # Fallback to enhanced local knowledge base
        return self._get_enhanced_fallback_response(question, target_language)
    
    def _get_enhanced_fallback_response(self, question: str, target_language: str) -> str:
        """Enhanced fallback responses with better multilingual support"""
        q = question.lower().strip()
        
        # Enhanced response templates in multiple languages
        responses = {
            'en': {
                'time': f"The current time is {datetime.now().strftime('%I:%M %p')}",
                'date': f"Today is {datetime.now().strftime('%A, %B %d, %Y')}",
                'weather': "I recommend checking your weather app for current conditions",
                'greeting': "Hello! How can I help you today?",
                'thanks': "You're welcome! Anything else I can help with?",
                'identity': "I'm your voice assistant, here to help you",
                'recipe': "For recipes like biryani, I can give basic steps. Would you like me to help?",
                'navigate': "To navigate, please switch to Navigation Mode and tell me your destination",
                'default': "I'm here to help! Could you ask that differently?"
            },
            'hi': {
                'time': f"अभी समय है {datetime.now().strftime('%I:%M %p')}",
                'date': f"आज है {datetime.now().strftime('%A, %d %B %Y')}",
                'weather': "मौसम की जानकारी के लिए मौसम ऐप देखें",
                'greeting': "नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूं?",
                'thanks': "आपका स्वागत है! कुछ और मदद चाहिए?",
                'identity': "मैं आपका वॉयस असिस्टेंट हूं, आपकी मदद के लिए यहां हूं",
                'recipe': "बिरयानी जैसी रेसिपी के लिए मैं बेसिक स्टेप्स बता सकता हूं",
                'navigate': "नेविगेशन के लिए नेविगेशन मोड में जाएं",
                'default': "मैं यहां मदद के लिए हूं! कृपया दूसरे तरीके से पूछें"
            },
            'te': {
                'time': f"ఇప్పుడు సమయం {datetime.now().strftime('%I:%M %p')}",
                'date': f"ఈరోజు {datetime.now().strftime('%A, %d %B %Y')}",
                'weather': "వాతావరణం కోసం వెదర్ యాప్ చూడండి",
                'greeting': "హలో! ఈరోజు నేను మీకు ఎలా సహాయం చేయగలను?",
                'thanks': "మీకు స్వాగతం! మరేదైనా సహాయం కావాలా?",
                'identity': "నేను మీ వాయిస్ అసిస్టెంట్, మీ సహాయానికి ఇక్కడ ఉన్నాను",
                'recipe': "బిర్యానీ వంటి వంటకాలకు నేను ప్రాథమిక దశలు చెప్పగలను",
                'navigate': "నేవిగేషన్ కోసం నేవిగేషన్ మోడ్‌కు వెళ్ళండి",
                'default': "నేను సహాయానికి ఇక్కడ ఉన్నాను! దయచేసి వేరే విధంగా అడగండి"
            },
            'es': {
                'time': f"La hora actual es {datetime.now().strftime('%I:%M %p')}",
                'date': f"Hoy es {datetime.now().strftime('%A, %d de %B de %Y')}",
                'weather': "Te recomiendo revisar tu app del clima",
                'greeting': "¡Hola! ¿Cómo puedo ayudarte hoy?",
                'thanks': "¡De nada! ¿Necesitas algo más?",
                'identity': "Soy tu asistente de voz, aquí para ayudarte",
                'recipe': "Para recetas como biryani, puedo dar pasos básicos",
                'navigate': "Para navegar, ve al Modo Navegación",
                'default': "¡Estoy aquí para ayudar! ¿Podrías preguntarlo de otra manera?"
            }
        }
        
        lang_responses = responses.get(target_language, responses['en'])
        
        # Enhanced question detection
        if any(word in q for word in ['time', 'clock', 'समय', 'సమయం', 'hora']):
            return lang_responses['time']
        elif any(word in q for word in ['date', 'today', 'day', 'आज', 'ఈరోజు', 'hoy']):
            return lang_responses['date']
        elif any(word in q for word in ['weather', 'temperature', 'मौसम', 'వాతావరణం', 'clima']):
            return lang_responses['weather']
        elif any(word in q for word in ['hello', 'hi', 'hey', 'नमस्ते', 'हैलो', 'హలో', 'hola']):
            return lang_responses['greeting']
        elif any(word in q for word in ['thank', 'thanks', 'धन्यवाद', 'థాంక్స్', 'gracias']):
            return lang_responses['thanks']
        elif any(word in q for word in ['who are you', 'what are you', 'आप कौन', 'మీరు ఎవరు', 'quién eres']):
            return lang_responses['identity']
        elif any(word in q for word in ['recipe', 'cook', 'biryani', 'रेसिपी', 'बिरयानी', 'వంటకం', 'బిర్యానీ', 'receta']):
            return lang_responses['recipe']
        elif any(word in q for word in ['navigate', 'directions', 'route', 'नेविगेशन', 'रास्ता', 'నేవిగేషన్', 'దిశలు', 'navegar']):
            return lang_responses['navigate']
        else:
            return lang_responses['default']

######################################################################
# Enhanced Voice Assistant with No File Saving and Faster Response
######################################################################
class VoiceAssistant:
    """Enhanced voice processing with in-memory audio and faster response"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize TTS engine once and keep it
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 160)  # Slightly faster speech
            self.tts_engine.setProperty('volume', 0.9)
            self.tts_initialized = True
        except:
            self.tts_initialized = False
        
        # Initialize pygame once
        try:
            pygame.mixer.quit()
            pygame.quit()
            pygame.init()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.pygame_initialized = True
        except:
            self.pygame_initialized = False
        
        # Multi-language support
        self.supported_languages = {
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'german': 'de',
            'italian': 'it',
            'portuguese': 'pt',
            'russian': 'ru',
            'chinese': 'zh',
            'japanese': 'ja',
            'korean': 'ko',
            'hindi': 'hi',
            'arabic': 'ar',
            'telugu': 'te'
        }
        
        self.current_language = 'en'
        
    def listen(self, timeout=8, phrase_time_limit=4):
        """Faster speech recognition with reduced timeouts"""
        try:
            with self.microphone as source:
                # Quick noise adjustment
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                st.info("Listening... Speak now!")
                
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
                
            # Faster recognition - try Google first with shorter timeout
            try:
                text = self.recognizer.recognize_google(audio, language=self.current_language)
                return text
            except:
                # Quick fallback to built-in recognition
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text
                except:
                    pass
                    
            return "Could not understand audio"
            
        except sr.WaitTimeoutError:
            return "Timeout: No speech detected"
        except Exception as e:
            return f"Error in speech recognition: {str(e)}"
    
    def speak(self, text: str, use_gtts=True, show_text=True):
        """Enhanced speech with in-memory audio processing and guaranteed first response"""
        if show_text:
            st.success(f"🔊 Assistant: {text}")
        
        # Always ensure we get audio output on first try
        success = False
        
        # Method 1: Try pyttsx3 first (works offline and faster)
        if self.tts_initialized and not success:
            try:
                # Force engine to speak immediately
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                success = True
            except Exception as e:
                try:
                    # Reinitialize engine if failed
                    self.tts_engine.stop()
                    self.tts_engine = pyttsx3.init()
                    self.tts_engine.setProperty('rate', 160)
                    self.tts_engine.setProperty('volume', 0.9)
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    success = True
                except:
                    pass
        
        # Method 2: Use gTTS with in-memory processing (no file saving)
        if not success and use_gtts and self.pygame_initialized:
            try:
                # Create gTTS object
                tts = gTTS(text=text, lang=self.current_language, slow=False)
                
                # Save to in-memory buffer instead of file
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                # Play from memory buffer
                sound = pygame.mixer.Sound(audio_buffer)
                channel = sound.play()
                
                # Wait for completion
                while channel.get_busy():
                    pygame.time.Clock().tick(10)
                
                success = True
                
            except Exception as e:
                pass
        
        # Method 3: Fallback - reinitialize everything and try once more
        if not success:
            try:
                # Reinitialize pygame
                pygame.mixer.quit()
                pygame.init()
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                
                # Create simple TTS
                tts = gTTS(text=text, lang='en', slow=False)  # Force English as last resort
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                sound = pygame.mixer.Sound(audio_buffer)
                sound.play()
                
                # Don't wait - just fire and continue
                success = True
            except:
                # If all else fails, at least show the text
                if not show_text:
                    st.info(f"🔊 {text}")

######################################################################
# Helper Functions (unchanged)
######################################################################

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    try:
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    except Exception:
        return 0.0

def get_current_precise_location():
    """Get more precise location using multiple services"""
    try:
        # Try ipgeolocation.io for better accuracy
        response = requests.get("https://ipgeolocation.io/json", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('latitude') and data.get('longitude'):
                return {
                    'latitude': float(data['latitude']),
                    'longitude': float(data['longitude']),
                    'address': f"{data.get('city', '')}, {data.get('state_prov', '')}, {data.get('country_name', '')}"
                }
    except:
        pass
    
    try:
        # Try ip-api.com (no API key required)
        response = requests.get("http://ip-api.com/json/?fields=status,message,country,regionName,city,lat,lon", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return {
                    'latitude': float(data['lat']),
                    'longitude': float(data['lon']),
                    'address': f"{data.get('city', '')}, {data.get('regionName', '')}, {data.get('country', '')}"
                }
    except:
        pass
    
    # Default location if all services fail
    return {
        'latitude': 17.3850,
        'longitude': 78.4867,
        'address': "Hyderabad, Telangana, India"
    }

def _normalize_lang(lang_code: str) -> str:
    """Normalize codes like 'pt-BR' -> 'pt' for some APIs."""
    if not lang_code:
        return 'en'
    return lang_code.split('-')[0]

def translate_text(text: str, target_lang: str) -> str:
    """Fast translation with immediate fallback - simplified for speed"""
    if not text or not target_lang or target_lang == 'en':
        return text
    
    # Very basic quick translations for common phrases only
    quick_translations = {
        'hi': {
            'Hello! How can I help you today?': 'नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूं?',
            'Navigation started': 'नेविगेशन शुरू हो गया',
            'Navigation ended': 'नेविगेशन समाप्त हो गया'
        },
        'te': {
            'Hello! How can I help you today?': 'హలో! ఈరోజు నేను మీకు ఎలా సహాయం చేయగలను?',
            'Navigation started': 'నేవిగేషన్ ప్రారంభమైంది',
            'Navigation ended': 'నేవిగేషన్ ముగిసింది'
        }
    }
    
    # Return quick translation if available, otherwise return original
    return quick_translations.get(target_lang, {}).get(text, text)

######################################################################
# GPS Navigator (unchanged)
######################################################################
class GPSNavigator:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="voice_assistant_enhanced_v1")
        self.current_location = None
        self.destination = None

    def get_current_location(self):
        """Get real-time precise current location"""
        self.current_location = get_current_precise_location()
        return self.current_location

    def geocode(self, query: str) -> Optional[Dict]:
        """Enhanced geocoding with multiple services"""
        query = query.strip()
        if not query:
            return None
        
        # Get current location for context
        current_loc = self.get_current_location()
        current_city = current_loc['address'].split(',')[0].strip()
        
        # Enhanced search queries with current location context
        search_queries = [
            query,
            f"{query}, {current_city}",
            f"{query}, India",
            f"{query} near {current_city}",
            f"{query} {current_city} India"
        ]
        
        # Try Nominatim with enhanced parameters
        for search_query in search_queries:
            try:
                location = self.geolocator.geocode(
                    search_query,
                    timeout=10,
                    exactly_one=True,
                    limit=1,
                    country_codes=['IN'],
                    addressdetails=True
                )
                if location:
                    return {
                        'latitude': location.latitude,
                        'longitude': location.longitude,
                        'address': location.address
                    }
                time.sleep(0.2)
            except Exception:
                continue
        
        return None

    def fetch_route_osrm(self, start_lat, start_lon, end_lat, end_lon, profile='driving'):
        """Enhanced route fetching with better routing services"""
        
        osrm_servers = [
            "https://router.project-osrm.org",
            "http://router.project-osrm.org"
        ]
        
        for server in osrm_servers:
            try:
                url = f"{server}/route/v1/{profile}/{start_lon},{start_lat};{end_lon},{end_lat}"
                params = {
                    'overview': 'full',
                    'geometries': 'geojson',
                    'steps': 'true',
                    'annotations': 'distance,duration'
                }
                
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('routes'):
                        route = data['routes'][0]
                        coords = route['geometry']['coordinates']
                        
                        steps = []
                        for leg in route.get('legs', []):
                            for step in leg.get('steps', []):
                                instruction = step.get('maneuver', {}).get('instruction', 'Continue')
                                if not instruction or instruction == 'Continue':
                                    maneuver_type = step.get('maneuver', {}).get('type', 'straight')
                                    if maneuver_type == 'turn':
                                        modifier = step.get('maneuver', {}).get('modifier', 'straight')
                                        instruction = f"Turn {modifier}"
                                    elif maneuver_type == 'depart':
                                        instruction = "Head out"
                                    elif maneuver_type == 'arrive':
                                        instruction = "You have arrived"
                                    else:
                                        instruction = f"{maneuver_type.replace('-', ' ').title()}"
                                
                                steps.append({
                                    'instruction': instruction,
                                    'distance': step.get('distance', 0),
                                    'duration': step.get('duration', 0)
                                })
                        
                        return coords, steps, route.get('distance', 0), route.get('duration', 0)
                        
            except Exception as e:
                continue
        
        # Fallback: create simple direct route
        coords = [[start_lon, start_lat], [end_lon, end_lat]]
        distance = haversine_m(start_lat, start_lon, end_lat, end_lon)
        duration = distance / (13 if profile == 'driving' else 1.4)
        
        steps = [{
            'instruction': f'Head towards destination ({distance/1000:.1f} km)',
            'distance': distance,
            'duration': duration
        }]
        
        return coords, steps, distance, duration

######################################################################
# Enhanced AI Assistant with Faster LLM Integration
######################################################################
class IntelligentAssistant:
    def __init__(self):
        self.voice = VoiceAssistant()
        self.nav = GPSNavigator()
        self.llm = LLMAssistant()
        self.current_mode = 'assistant'
        self.emergency_contacts = {
            'police': '100',
            'fire': '101', 
            'ambulance': '102',
            'emergency': '112'
        }

    def process_voice_command(self, command: str):
        """Enhanced command processing with faster LLM integration"""
        if not command or len(command.strip()) < 2:
            target_lang = _normalize_lang(self.voice.current_language)
            return translate_text("I didn't catch that. Please try again.", target_lang)
        
        cl = command.lower().strip()
        target_lang = _normalize_lang(self.voice.current_language)
        
        # Emergency detection (immediate response)
        if any(word in cl for word in ['emergency', 'help', 'urgent', 'danger', 'police', 'fire', 'ambulance']):
            return self.handle_emergency(command)
        
        # Navigation detection (brief response for navigation panel)
        if any(phrase in cl for phrase in ['navigate to', 'directions to', 'route to', 'go to', 'take me to']):
            return translate_text("Switch to Navigation Mode for directions", target_lang)
        
        # Use LLM for all other queries with faster timeout
        try:
            response = self.llm.get_llm_response(command, target_lang)
            return response
        except Exception as e:
            # Quick fallback
            return translate_text("I'm having trouble right now. Please try again.", target_lang)

    def handle_emergency(self, command: str) -> str:
        """Compact emergency handler"""
        emergency_type = command.lower()
        target_lang = _normalize_lang(self.voice.current_language)
        
        if any(word in emergency_type for word in ['police', 'crime']):
            response = f"🚨 POLICE: Dial {self.emergency_contacts['police']} immediately"
        elif any(word in emergency_type for word in ['fire', 'burning']):
            response = f"🚨 FIRE: Dial {self.emergency_contacts['fire']} immediately"
        elif any(word in emergency_type for word in ['medical', 'ambulance']):
            response = f"🚨 MEDICAL: Dial {self.emergency_contacts['ambulance']} immediately"
        else:
            response = f"🚨 EMERGENCY: Dial {self.emergency_contacts['emergency']} immediately"
        
        return translate_text(response, target_lang)

######################################################################
# Main Streamlit App (same interface, enhanced functionality)
######################################################################
def main():
    st.set_page_config(
        page_title="Advanced Voice Assistant", 
        page_icon="🤖", 
        layout="wide"
    )

    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = IntelligentAssistant()
    if 'nav_session' not in st.session_state:
        st.session_state.nav_session = None
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = 'assistant'
    if 'user_moved' not in st.session_state:
        st.session_state.user_moved = True

    app = st.session_state.assistant

    # Header with mode switching
    st.title("🤖 Advanced Multilingual Voice Assistant")
    
    # Mode selection buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Assistant Mode", use_container_width=True, type="primary" if st.session_state.current_mode == 'assistant' else "secondary"):
            st.session_state.current_mode = 'assistant'
            st.rerun()
    
    with col2:
        if st.button("🗺️ Navigation Mode", use_container_width=True, type="primary" if st.session_state.current_mode == 'navigation' else "secondary"):
            st.session_state.current_mode = 'navigation'
            st.rerun()
    
    with col3:
        if st.button("🚨 Emergency Mode", use_container_width=True, type="primary" if st.session_state.current_mode == 'emergency' else "secondary"):
            st.session_state.current_mode = 'emergency'
            st.rerun()

    st.markdown("---")

    # Language settings sidebar
    with st.sidebar:
        st.header("🌐 Settings")
        lang_name = st.selectbox("Language:", list(app.voice.supported_languages.keys()), index=0)
        app.voice.current_language = app.voice.supported_languages.get(lang_name, 'en')
        
        st.markdown("---")
        st.subheader("📍 Current Location")
        current_loc = app.nav.get_current_location()
        st.write(f"**Address:** {current_loc['address']}")
        st.write(f"**Coordinates:** {current_loc['latitude']:.4f}, {current_loc['longitude']:.4f}")

    # MODE-SPECIFIC PAGES
    
    #############################
    # ASSISTANT MODE
    #############################
    if st.session_state.current_mode == 'assistant':
        st.header("💬 AI Assistant")
        st.write("Ask me anything! I can help with general knowledge, science, history, current time, recipes, and much more.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Voice input
            st.subheader("🎤 Voice Interaction")
            if st.button("🎤 Start Listening", key="assistant_listen", use_container_width=True):
                with st.spinner("Listening for your question..."):
                    command = app.voice.listen()
                
                if command and not any(err in command for err in ["Error", "Network error", "Could not understand", "No speech", "Timeout"]):
                    st.info(f"**You asked:** {command}")
                    
                    with st.spinner("Thinking..."):
                        response = app.process_voice_command(command)
                    
                    # Show text response and speak (in selected language)
                    app.voice.speak(response, show_text=True)
                else:
                    st.warning(f"⚠️ {command}")
            
            # Text input alternative
            st.subheader("⌨️ Text Input")
            text_question = st.text_input("Type your question here:", placeholder="Ask me anything...")
            
            if st.button("📤 Ask Question", use_container_width=True) and text_question:
                st.info(f"**You asked:** {text_question}")
                
                with st.spinner("Thinking..."):
                    response = app.process_voice_command(text_question)
                
                app.voice.speak(response, show_text=True)
        
        with col2:
            st.subheader("💡 Sample Questions")
            sample_questions = [
                "What time is it?",
                "What's today's date?", 
                "How to make biryani?",
                "Who was APJ Abdul Kalam?",
                "Explain photosynthesis",
                "Tell me about Indian history",
                "How does the internet work?",
                "What is the capital of France?"
            ]
            
            for q in sample_questions:
                if st.button(f"📘 {q}", key=f"sample_{hash(q)}", use_container_width=True):
                    st.info(f"**You asked:** {q}")
                    response = app.process_voice_command(q)
                    app.voice.speak(response, show_text=True)

    #############################
    # NAVIGATION MODE  
    #############################
    elif st.session_state.current_mode == 'navigation':
        st.header("🗺️ Navigation System")
        st.write("Get precise directions to any location with voice guidance.")
        
        # Voice input for navigation
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎤 Voice Navigation Input")
            if st.button("🎤 Say Destination", key="nav_voice", use_container_width=True):
                with st.spinner("Listening for destination..."):
                    voice_input = app.voice.listen()
                
                if voice_input and not any(err in voice_input for err in ["Error", "Network error", "Could not understand", "No speech", "Timeout"]):
                    st.info(f"**You said:** {voice_input}")
                    
                    destination = voice_input.lower().strip()
                    nav_keywords = ['navigate to', 'directions to', 'route to', 'go to', 'take me to', 'drive to', 'walk to']
                    for keyword in nav_keywords:
                        if keyword in destination:
                            destination = destination.split(keyword, 1)[-1].strip()
                            break
                    
                    if len(destination) > 2:
                        with st.spinner("Calculating route..."):
                            result = start_navigation_process(app, destination, "driving")
                            if result['success']:
                                # Brief response for navigation panel
                                target_lang = _normalize_lang(app.voice.current_language)
                                brief_msg = translate_text("Navigation started", target_lang)
                                app.voice.speak(brief_msg, show_text=True)
                                st.rerun()
                            else:
                                st.error(result['message'])
                                app.voice.speak(translate_text("Destination not found", _normalize_lang(app.voice.current_language)), show_text=False)
                    else:
                        msg = "Please specify a destination"
                        st.warning(msg)
                        app.voice.speak(translate_text(msg, _normalize_lang(app.voice.current_language)), show_text=False)
                else:
                    st.warning(f"⚠️ {voice_input}")
        
        with col2:
            transport_mode = st.selectbox("🚗 Mode:", ["driving", "walking"], key="nav_mode")
        
        # Text input for navigation
        st.subheader("⌨️ Text Input Navigation")
        destination = st.text_input("🎯 Enter destination:", placeholder="e.g., Charminar Hyderabad, Red Fort Delhi, Gateway of India Mumbai")
        
        if st.button("🧭 Start Navigation", use_container_width=True) and destination:
            result = start_navigation_process(app, destination, transport_mode)
            if result['success']:
                target_lang = _normalize_lang(app.voice.current_language)
                brief_msg = translate_text("Navigation started", target_lang)
                app.voice.speak(brief_msg, show_text=True)
                st.rerun()
            else:
                st.error(result['message'])
        
        # Quick destination buttons for testing
        st.subheader("📘 Quick Destinations")
        quick_destinations = [
            "Charminar Hyderabad",
            "Red Fort Delhi", 
            "Gateway of India Mumbai",
            "Mysore Palace",
            "Golden Temple Amritsar"
        ]
        
        cols = st.columns(3)
        for i, dest in enumerate(quick_destinations):
            with cols[i % 3]:
                if st.button(f"📍 {dest}", key=f"quick_{i}", use_container_width=True):
                    result = start_navigation_process(app, dest, transport_mode)
                    if result['success']:
                        target_lang = _normalize_lang(app.voice.current_language)
                        brief_msg = translate_text("Navigation started", target_lang)
                        app.voice.speak(brief_msg, show_text=True)
                        st.rerun()
        
        # Display active navigation
        if st.session_state.nav_session and st.session_state.nav_session.get('active'):
            display_enhanced_navigation(app)
        else:
            st.info("🗺️ No active navigation. Use voice command or enter destination above to start navigation.")

    #############################
    # EMERGENCY MODE
    #############################
    elif st.session_state.current_mode == 'emergency':
        st.header("🚨 Emergency Assistance")
        st.error("EMERGENCY MODE ACTIVATED")
        
        # Emergency contacts display
        st.subheader("📞 Emergency Contacts (India)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🚓 Police", "100")
            st.metric("🚒 Fire Department", "101")
        
        with col2:
            st.metric("🚑 Ambulance", "102") 
            st.metric("📞 Universal Emergency", "112")
        
        st.markdown("---")
        
        # Emergency voice input
        st.subheader("🎤 Describe Your Emergency")
        
        if st.button("🚨 EMERGENCY VOICE INPUT", use_container_width=True, type="primary"):
            with st.spinner("Listening to emergency..."):
                command = app.voice.listen()
            
            if command and not any(err in command for err in ["Error", "Network error", "Could not understand", "Timeout"]):
                st.warning(f"**Emergency reported:** {command}")
                
                response = app.handle_emergency(command)
                st.error(response)
                app.voice.speak(response, show_text=False)
            else:
                msg = "Could not process emergency input. Please try again or dial emergency services directly."
                st.warning(msg)
                target_lang = _normalize_lang(app.voice.current_language)
                app.voice.speak(translate_text(msg, target_lang), show_text=False)
        
        # Quick emergency buttons
        st.subheader("📘 Quick Emergency Actions")
        
        emergency_types = [
            ("🚓 Police Emergency", "police emergency"),
            ("🚒 Fire Emergency", "fire emergency"), 
            ("🚑 Medical Emergency", "medical emergency"),
            ("🆘 General Emergency", "general emergency")
        ]
        
        for label, emergency_type in emergency_types:
            if st.button(label, use_container_width=True):
                response = app.handle_emergency(emergency_type)
                st.error(response)
                app.voice.speak(response, show_text=False)
        
        # Current location for emergency services
        st.subheader("📍 Your Location (for emergency services)")
        current_loc = app.nav.get_current_location()
        st.code(f"""
Address: {current_loc['address']}
Coordinates: {current_loc['latitude']:.6f}, {current_loc['longitude']:.6f}
        """)
        
        st.warning("⚠️ **IMPORTANT:** In a real emergency, call emergency services immediately. This app is for assistance only.")

# Enhanced navigation processing with real location
def start_navigation_process(app, destination, transport_mode):
    """Enhanced navigation with real-time location and brief responses"""
    try:
        # Get precise current location
        current_loc = app.nav.get_current_location()
        
        # Enhanced geocoding with better error handling
        with st.spinner(f"Finding {destination}..."):
            dest_loc = app.nav.geocode(destination)
        
        if not dest_loc:
            # Try alternative search strategies
            alt_searches = [
                f"{destination} landmark India",
                f"{destination} tourist place India", 
                f"{destination} monument India",
                destination.replace(" ", "+")
            ]
            
            for alt_search in alt_searches:
                dest_loc = app.nav.geocode(alt_search)
                if dest_loc:
                    break
        
        if not dest_loc:
            return {
                'success': False,
                'message': f"Could not find '{destination}'. Please try with more specific details."
            }
        
        # Calculate route with enhanced error handling
        with st.spinner("Calculating route..."):
            coords, steps, distance_m, duration_s = app.nav.fetch_route_osrm(
                current_loc['latitude'], current_loc['longitude'],
                dest_loc['latitude'], dest_loc['longitude'],
                transport_mode
            )
        
        # Create enhanced navigation session
        st.session_state.nav_session = {
            'active': True,
            'coords': coords,
            'steps': steps,
            'current_step': 0,
            'current_position': 0,
            'destination': dest_loc,
            'start_location': current_loc,
            'distance_m': distance_m,
            'duration_s': duration_s,
            'last_instruction_time': 0,
            'spoken_steps': set(),
            'transport_mode': transport_mode,
            'last_update': time.time(),
            'total_steps': len(steps),
            'route_started': time.time()
        }
        
        # Brief success message
        distance_km = distance_m / 1000
        first_instruction = steps[0]['instruction'] if steps else "Head towards destination"
        
        message = f"Route found: {distance_km:.1f}km. {first_instruction}"
        
        return {
            'success': True,
            'message': message
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f"Navigation failed. Please check connection and try again."
        }

# Simplified navigation display with brief voice instructions
def display_enhanced_navigation(app):
    """Enhanced navigation with brief voice instructions"""
    nav = st.session_state.nav_session
    
    # Real-time navigation header - simplified
    if nav['current_step'] < len(nav['steps']):
        current_instruction = nav['steps'][nav['current_step']]['instruction']
        st.success(f"🧭 **Now:** {current_instruction}")
        
        if nav['current_step'] + 1 < len(nav['steps']):
            next_instruction = nav['steps'][nav['current_step'] + 1]['instruction']
            st.info(f"⭐ **Next:** {next_instruction}")
    
    # Enhanced interactive map (same as before)
    current_loc = nav['start_location']
    coords = nav['coords']
    
    # Calculate current position based on time elapsed and progress
    time_elapsed = time.time() - nav['route_started']
    progress_ratio = min(time_elapsed / nav['duration_s'], 1.0) if nav['duration_s'] > 0 else 0
    nav['current_position'] = int(progress_ratio * len(coords))
    
    # Center map on current position
    if nav['current_position'] < len(coords):
        center_lon, center_lat = coords[nav['current_position']]
    else:
        center_lat = current_loc['latitude']
        center_lon = current_loc['longitude']
    
    # Create enhanced map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles=None
    )
    
    # Add map layers
    folium.TileLayer('OpenStreetMap', name='Standard').add_to(m)
    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Enhanced route display
    route_points = [(lat, lon) for lon, lat in coords]
    
    # Main route (thick blue line)
    folium.PolyLine(
        route_points,
        weight=8,
        color='#1E88E5',
        opacity=0.9,
        tooltip=f"Route to {nav['destination']['address']}"
    ).add_to(m)
    
    # Progress indicator
    if nav['current_position'] > 0:
        completed_points = route_points[:nav['current_position']+1]
        folium.PolyLine(
            completed_points,
            weight=8,
            color='#4CAF50',
            opacity=0.9,
            tooltip="Completed route"
        ).add_to(m)
    
    # Markers
    start_loc = nav['start_location']
    folium.Marker(
        [start_loc['latitude'], start_loc['longitude']],
        popup=f"Start: {start_loc['address']}",
        tooltip="Starting Point 🚩",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    dest = nav['destination']
    folium.Marker(
        [dest['latitude'], dest['longitude']],
        popup=f"Destination: {dest['address']}",
        tooltip="Destination 🏁",
        icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa')
    ).add_to(m)
    
    # Current position marker
    if nav['current_position'] < len(coords):
        current_lon, current_lat = coords[nav['current_position']]
        
        folium.CircleMarker(
            [current_lat, current_lon],
            radius=15,
            popup="Your Current Position",
            tooltip="You are here 🔵",
            color='white',
            weight=4,
            fill=True,
            fillColor='#2196F3',
            fillOpacity=1.0
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display the map
    map_data = st_folium(
        m, 
        width=900, 
        height=500,  # Reduced height for better UI
        returned_objects=["last_object_clicked", "bounds"],
        key="nav_map"
    )
    
    # Navigation controls
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🛑 End Navigation", use_container_width=True):
            st.session_state.nav_session = None
            st.success("Navigation ended")
            target_lang = _normalize_lang(app.voice.current_language)
            app.voice.speak(translate_text("Navigation ended", target_lang), show_text=False)
            st.rerun()
    
    with col2:
        total_distance = nav['distance_m']
        progress = min(nav['current_position'] / len(coords), 1.0) if len(coords) > 0 else 0
        remaining_distance = total_distance * (1 - progress)
        
        if remaining_distance > 1000:
            st.metric("Remaining", f"{remaining_distance/1000:.1f} km")
        else:
            st.metric("Remaining", f"{remaining_distance:.0f} m")
    
    with col3:
        speed = 50 if nav['transport_mode'] == 'driving' else 5
        remaining_time = (remaining_distance / 1000) / speed * 60
        
        if remaining_time < 60:
            st.metric("ETA", f"{remaining_time:.0f} min")
        else:
            hours = int(remaining_time // 60)
            mins = int(remaining_time % 60)
            st.metric("ETA", f"{hours}h {mins}m")
    
    with col4:
        progress_pct = progress * 100
        st.metric("Progress", f"{progress_pct:.0f}%")
    
    with col5:
        elapsed_min = (time.time() - nav['route_started']) / 60
        if elapsed_min < 60:
            st.metric("Elapsed", f"{elapsed_min:.0f} min")
        else:
            hours = int(elapsed_min // 60)
            mins = int(elapsed_min % 60)
            st.metric("Elapsed", f"{hours}h {mins}m")
    
    # Brief voice guidance - only essential instructions
    current_time = time.time()
    
    # Provide brief instructions every 45 seconds
    if (current_time - nav.get('last_instruction_time', 0) > 45 and 
        nav['current_step'] not in nav['spoken_steps']):
        
        if nav['current_step'] < len(nav['steps']):
            step = nav['steps'][nav['current_step']]
            instruction = step['instruction']
            
            # Make instruction brief for voice
            brief_instruction = instruction.split('.')[0]  # Take first sentence only
            
            # Voice announcement in user's language - brief version
            target_lang = _normalize_lang(app.voice.current_language)
            translated_instruction = translate_text(brief_instruction, target_lang)
            
            app.voice.speak(translated_instruction, show_text=False)
            
            nav['spoken_steps'].add(nav['current_step'])
            nav['last_instruction_time'] = current_time
            
            # Move to next step based on progress
            if progress > (nav['current_step'] + 1) / len(nav['steps']):
                nav['current_step'] += 1
    
    # Check if arrived at destination
    if progress >= 0.95:
        if nav.get('active', True):
            arrival_msg = "Arrived at destination"
            st.balloons()
            st.success(f"🎉 {arrival_msg}")
            
            target_lang = _normalize_lang(app.voice.current_language)
            app.voice.speak(translate_text(arrival_msg, target_lang), show_text=False)
            
            nav['active'] = False
            time.sleep(2)
            st.session_state.nav_session = None
            st.rerun()
    
    # Reduced refresh rate for better performance
    if nav.get('active', True):
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()