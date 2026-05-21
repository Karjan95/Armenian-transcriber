import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({apiKey: process.env.GEMINI_API_KEY });
async function test() {
  try {
    let r1 = await ai.models.generateContent({ model: 'gemini-1.5-pro', contents: 'hello' });
    console.log("1.5 worked");
  } catch(e) {}
  try {
    let r2 = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: 'hello' });
    console.log("2.5 flash worked");
  } catch(e) { console.log(e.message); }
  try {
    let r3 = await ai.models.generateContent({ model: 'gemini-2.5-pro', contents: 'hello' });
    console.log("2.5 pro worked");
  } catch(e) { console.log(e.message); }
}
test();
