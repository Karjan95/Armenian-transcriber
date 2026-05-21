import { GoogleGenAI } from '@google/genai';
import { readFileSync, writeFileSync } from 'fs';

async function run() {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || 'fake' });
  try {
    let resp = await ai.models.generateContent({
        model: 'gemini-3.1-flash-preview',
        contents: 'hello'
    });
    console.log('3.1 flash:', resp.text);
  } catch (err) {
    console.error('ERROR:', err);
  }
}
run();
