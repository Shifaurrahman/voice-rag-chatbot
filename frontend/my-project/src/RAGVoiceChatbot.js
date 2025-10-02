import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Send, MessageCircle, Loader2, AlertCircle, CheckCircle, Mic, MicOff, Volume2, StopCircle } from 'lucide-react';

const RAGVoiceChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiHealth, setApiHealth] = useState(null);
  const [topK, setTopK] = useState(3);
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  
  // Voice-related states
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState('alloy');
  const [recordingTime, setRecordingTime] = useState(0);
  const [micPermission, setMicPermission] = useState('prompt'); // 'granted', 'denied', 'prompt'
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingIntervalRef = useRef(null);
  const messagesEndRef = useRef(null);

  const voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'];

  const checkApiHealth = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/health`);
      const data = await response.json();
      setApiHealth(data);
    } catch (error) {
      console.error('Health check failed:', error);
      setApiHealth({ status: 'error', message: 'Cannot connect to API' });
    }
  }, [apiUrl]);

  useEffect(() => {
    checkApiHealth();
    checkMicrophonePermission();
  }, [checkApiHealth]);

  const checkMicrophonePermission = async () => {
    try {
      const result = await navigator.permissions.query({ name: 'microphone' });
      setMicPermission(result.state);
      
      result.onchange = () => {
        setMicPermission(result.state);
      };
    } catch (error) {
      console.log('Permission API not supported');
      setMicPermission('prompt');
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
      isVoice: false
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(`${apiUrl}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,
          top_k: topK
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: data.answer,
        timestamp: new Date(),
        isVoice: false,
        metadata: {
          question: data.question,
          sources: data.sources || [],
          confidence: data.confidence || 0,
          processingTime: data.processing_time_ms || 0,
          totalSources: data.total_sources || 0
        }
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: `Error: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setMicPermission('granted');
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await sendVoiceMessage(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);

      // Start timer
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Error accessing microphone:', error);
      setMicPermission('denied');
      alert('Could not access microphone. Please check your permissions in browser settings.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(recordingIntervalRef.current);
      setRecordingTime(0);
    }
  };

  const sendVoiceMessage = async (audioBlob) => {
    setIsProcessingVoice(true);

    // Create audio URL for user's recording
    const userAudioUrl = URL.createObjectURL(audioBlob);

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: '🎤 Voice message (transcribing...)',
      timestamp: new Date(),
      isVoice: true,
      audioUrl: userAudioUrl // Store user's audio
    };

    setMessages(prev => [...prev, userMessage]);

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');

      const response = await fetch(`${apiUrl}/ask-voice?top_k=${topK}&voice=${selectedVoice}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Get transcription from headers
      const transcribedQuery = response.headers.get('X-Transcribed-Query');
      const answerText = response.headers.get('X-Answer-Text');

      // Update user message with transcription (keep audio URL)
      setMessages(prev => prev.map(msg => 
        msg.id === userMessage.id 
          ? { ...msg, content: `🎤 "${transcribedQuery}"` }
          : msg
      ));

      // Get audio response
      const audioResponseBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioResponseBlob);

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: answerText || 'Audio response received',
        timestamp: new Date(),
        isVoice: true,
        audioUrl: audioUrl,
        metadata: {
          question: transcribedQuery
        }
      };

      setMessages(prev => [...prev, botMessage]);

      // Auto-play the response
      const audio = new Audio(audioUrl);
      audio.play().catch(err => console.error('Error playing audio:', err));

    } catch (error) {
      console.error('Error processing voice:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: `Voice processing error: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessingVoice(false);
    }
  };

  const playAudio = (audioUrl) => {
    const audio = new Audio(audioUrl);
    audio.play().catch(err => console.error('Error playing audio:', err));
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatRecordingTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto max-w-4xl h-screen flex flex-col">
        {/* Header */}
        <div className="bg-white shadow-md rounded-t-lg mt-4 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <MessageCircle className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-800">RAG Chatbot</h1>
                <p className="text-sm text-gray-500">Ask questions with text or voice</p>
              </div>
            </div>
            
            {/* API Health Status */}
            <div className="flex items-center space-x-2">
              {apiHealth?.status === 'healthy' ? (
                <div className="flex items-center space-x-1 text-green-600">
                  <CheckCircle className="w-4 h-4" />
                  <span className="text-sm">API Online</span>
                  {apiHealth.voice_enabled && (
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded ml-2">
                      Voice ✓
                    </span>
                  )}
                </div>
              ) : (
                <div className="flex items-center space-x-1 text-red-600">
                  <AlertCircle className="w-4 h-4" />
                  <span className="text-sm">API Offline</span>
                </div>
              )}
            </div>
          </div>

          {/* Settings */}
          <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-200">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium text-gray-700">Top-K:</label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value={1}>1</option>
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                </select>
              </div>

              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium text-gray-700">Voice:</label>
                <select
                  value={selectedVoice}
                  onChange={(e) => setSelectedVoice(e.target.value)}
                  className="border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {voices.map(voice => (
                    <option key={voice} value={voice}>{voice}</option>
                  ))}
                </select>
              </div>
              
              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium text-gray-700">API:</label>
                <input
                  type="text"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  className="border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 w-40"
                  placeholder="http://localhost:8000"
                />
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={checkApiHealth}
                className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
              >
                Check API
              </button>
              <button
                onClick={clearChat}
                className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
              >
                Clear Chat
              </button>
            </div>
          </div>

          {/* API Info */}
          {apiHealth?.status === 'healthy' && (
            <div className="mt-2 text-xs text-gray-500">
              Documents: {apiHealth.total_documents} | Method: {apiHealth.embedding_method}
            </div>
          )}
        </div>

        {/* Chat Messages */}
        <div className="flex-1 bg-white px-6 py-4 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <MessageCircle className="w-16 h-16 mb-4 opacity-20" />
              <p className="text-lg mb-2">Welcome to RAG Chatbot!</p>
              <p className="text-sm text-center max-w-md">
                Ask questions using text or voice. I'll search through your documents and provide answers with citations.
              </p>
              <div className="mt-4 flex items-center space-x-2 text-xs text-gray-400">
                <Mic className="w-4 h-4" />
                <span>Voice input supported</span>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => (
                <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-3xl rounded-lg px-4 py-3 ${
                    message.type === 'user' 
                      ? 'bg-blue-600 text-white' 
                      : message.type === 'error'
                      ? 'bg-red-100 text-red-800 border border-red-200'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    
                    {/* Audio player for voice responses */}
                    {message.audioUrl && (
                      <div className="mt-2">
                        <button
                          onClick={() => playAudio(message.audioUrl)}
                          className="flex items-center space-x-2 bg-white bg-opacity-20 hover:bg-opacity-30 px-3 py-1.5 rounded text-sm transition-colors"
                        >
                          <Volume2 className="w-4 h-4" />
                          <span>Play Audio Response</span>
                        </button>
                      </div>
                    )}
                    
                    <div className={`text-xs mt-2 opacity-70 ${message.type === 'user' ? 'text-blue-100' : 'text-gray-500'}`}>
                      {formatTime(message.timestamp)}
                      {message.metadata && (
                        <span className="ml-2">
                          {message.metadata.processingTime && `• ${message.metadata.processingTime}ms`}
                          {message.metadata.totalSources > 0 && ` • Sources: ${message.metadata.totalSources}`}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              
              {(isLoading || isProcessingVoice) && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 rounded-lg px-4 py-3 flex items-center space-x-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-gray-600">
                      {isProcessingVoice ? 'Processing voice...' : 'Searching documents...'}
                    </span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 px-6 py-4 rounded-b-lg mb-4">
          {/* Recording indicator */}
          {isRecording && (
            <div className="mb-3 flex items-center justify-center space-x-2 text-red-600 animate-pulse">
              <div className="w-3 h-3 bg-red-600 rounded-full"></div>
              <span className="font-medium">Recording: {formatRecordingTime(recordingTime)}</span>
            </div>
          )}
          
          <div className="flex space-x-3">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !isRecording && sendMessage()}
              placeholder="Ask a question about your documents..."
              className="flex-1 border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading || isRecording || apiHealth?.status !== 'healthy'}
            />
            
            {/* Voice button */}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isProcessingVoice || apiHealth?.status !== 'healthy' || micPermission === 'denied'}
              className={`${
                isRecording 
                  ? 'bg-red-600 hover:bg-red-700' 
                  : micPermission === 'denied'
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-purple-600 hover:bg-purple-700'
              } text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 transition-colors`}
              title={
                micPermission === 'denied' 
                  ? 'Microphone access denied' 
                  : isRecording 
                  ? 'Stop recording' 
                  : 'Start voice input'
              }
            >
              {isRecording ? (
                <>
                  <StopCircle className="w-5 h-5" />
                  <span>Stop</span>
                </>
              ) : micPermission === 'denied' ? (
                <>
                  <MicOff className="w-5 h-5" />
                  <span>Denied</span>
                </>
              ) : (
                <>
                  <Mic className="w-5 h-5" />
                  <span>Voice</span>
                </>
              )}
            </button>
            
            {/* Send button */}
            <button
              onClick={sendMessage}
              disabled={isLoading || isRecording || !inputValue.trim() || apiHealth?.status !== 'healthy'}
              className="bg-blue-600 text-white rounded-lg px-6 py-3 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 transition-colors"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              <span>{isLoading ? 'Sending...' : 'Send'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAGVoiceChatbot;