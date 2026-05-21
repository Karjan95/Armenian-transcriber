import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({apiKey: "fake"});
async function test() {
  try {
    await ai.models.generateContent({
        model: 'gemini-9.9-fake-model',
        contents: 'hello'
    });
  } catch (e) {
    console.log(e.status, e.message);
  }
}
test();
