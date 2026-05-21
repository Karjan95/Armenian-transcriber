import { GoogleGenAI } from '@google/genai';

export async function testUpload(apiKey) {
  const ai = new GoogleGenAI({ apiKey });
  const blob = new Blob(['hello world'], { type: 'text/plain' });
  try {
    const uploadedFile = await ai.files.upload({
      file: blob,
      config: { mimeType: 'text/plain', displayName: 'test.txt' }
    });
    console.log('UPLOAD FILE:', uploadedFile);
    return uploadedFile;
  } catch (e) {
    console.error('UPLOAD ERROR:', e);
    throw e;
  }
}
