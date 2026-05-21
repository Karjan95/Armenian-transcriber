import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({apiKey: "fake"});
async function test() {
  try {
    await ai.files.get({ name: 'files/missing-file-123' });
  } catch (e) {
    console.log(e.status, e.message);
  }
}
test();
