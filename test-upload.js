import { GoogleGenAI } from '@google/genai';

async function testUpload(apiKey) {
  const ai = new GoogleGenAI({ apiKey });
  const blob = new Blob(['hello world'], { type: 'text/plain' });
  try {
    const uploadedFile = await ai.files.upload({
      file: blob,
      config: { mimeType: 'text/plain', displayName: 'test.txt' }
    });
    console.log('UPLOAD FILE:', uploadedFile);
    
    let isReady = false;
    while (!isReady) {
        const pollFile = await ai.files.get({ name: uploadedFile.name });
        console.log("POLL =>", pollFile.state);
        if (pollFile.state === 'ACTIVE') {
            isReady = true;
        } else if (pollFile.state === 'FAILED') {
            throw new Error("Cloud indexing failed.");
        } else {
            await new Promise(r => setTimeout(r, 2000));
        }
      }
      
  } catch (e) {
    if (e.response) {
      console.error('UPLOAD ERROR:', e.status, await e.response.text());
    } else {
      console.error('UPLOAD ERROR:', e);
    }
    throw e;
  }
}
testUpload('fake');
