import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({apiKey: "xxx"});
console.log(ai.apiClient.uploadFile.toString());
