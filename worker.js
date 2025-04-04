// Cloudflare AI Worker for OCR and Depression Analysis

addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request))
  })
  
  async function handleRequest(request) {
    // Only process POST requests with images
    if (request.method !== 'POST') {
      return new Response('Please send a POST request with an image', { status: 400 })
    }
  
    try {
      // Parse the form data to get the image
      const formData = await request.formData()
      const imageFile = formData.get('image')
      
      if (!imageFile) {
        return new Response('No image provided', { status: 400 })
      }
  
      // 1. Use Cloudflare's OCR model to extract text from the image
      const ocrResponse = await fetch(`https://api.cloudflare.com/client/v4/accounts/${ENV.ACCOUNT_ID}/ai/run/@cf/facebook/ocr-v3`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${ENV.API_TOKEN}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          image: await blobToBase64(imageFile)
        })
      })
  
      const ocrResult = await ocrResponse.json()
      const extractedText = ocrResult.result.text
      
      // 2. Perform depression sentiment analysis on the extracted text
      const sentimentScore = await analyzeSentimentForDepression(extractedText)
      
      return new Response(JSON.stringify({
        extractedText: extractedText,
        depressionScore: sentimentScore,
        scoreOutOf25: sentimentScore,
        interpretations: getScoreInterpretation(sentimentScore)
      }), {
        headers: { 'Content-Type': 'application/json' }
      })
    } catch (error) {
      return new Response(`Error processing request: ${error.message}`, { status: 500 })
    }
  }
  
  // Helper function to convert blob to base64
  async function blobToBase64(blob) {
    const arrayBuffer = await blob.arrayBuffer()
    const bytes = new Uint8Array(arrayBuffer)
    let binary = ''
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i])
    }
    return btoa(binary)
  }
  
  // Function to analyze text for depression markers
  async function analyzeSentimentForDepression(text) {
    // First use Cloudflare's text-classification model for sentiment
    const sentimentResponse = await fetch(`https://api.cloudflare.com/client/v4/accounts/${ENV.ACCOUNT_ID}/ai/run/@cf/huggingface/distilbert-sst-2-int8`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${ENV.API_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    })
    
    const sentimentResult = await sentimentResponse.json()
    const generalSentiment = sentimentResult.result
  
    // Then use a more specialized model for depression analysis
    // Here we're using Cloudflare's embedding model and then applying custom logic
    const embeddingResponse = await fetch(`https://api.cloudflare.com/client/v4/accounts/${ENV.ACCOUNT_ID}/ai/run/@cf/baai/bge-small-en-v1.5`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${ENV.API_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    })
    
    const embeddingResult = await embeddingResponse.json()
    
    // Custom logic to analyze depression markers in the text
    const depressionScore = calculateDepressionScore(text, generalSentiment, embeddingResult)
    
    return depressionScore
  }
  
  // Calculate depression score based on text content
  function calculateDepressionScore(text, sentiment, embedding) {
    // Initialize score
    let score = 12.5 // Start at middle point of 25-point scale
    
    // Adjust based on general sentiment
    if (sentiment.label === "NEGATIVE") {
      score += sentiment.score * 5
    } else {
      score -= sentiment.score * 5
    }
    
    // Key depression indicators - check for presence of these patterns
    const depressionIndicators = [
      { regex: /hopeless|worthless|emptiness|despair/gi, weight: 1.5 },
      { regex: /sad|down|low|blue|unhappy/gi, weight: 0.8 },
      { regex: /tired|exhausted|fatigue|no energy/gi, weight: 0.7 },
      { regex: /alone|lonely|isolated/gi, weight: 0.9 },
      { regex: /can't sleep|insomnia|sleeping too much/gi, weight: 0.6 },
      { regex: /no interest|don't care|apathy/gi, weight: 1.2 },
      { regex: /suicide|death|dying|end it/gi, weight: 2.5 },
      { regex: /guilt|blame|fault|shame/gi, weight: 1.0 },
      { regex: /can't concentrate|foggy|unfocused/gi, weight: 0.5 },
      { regex: /too slow|agitated|restless/gi, weight: 0.5 }
    ]
    
    // Check for depression indicators and adjust score
    depressionIndicators.forEach(indicator => {
      const matches = text.match(indicator.regex) || []
      score += matches.length * indicator.weight
    })
    
    // Check for positive indicators that may reduce depression score
    const positiveIndicators = [
      { regex: /happy|joy|grateful|thankful/gi, weight: -1.0 },
      { regex: /hopeful|looking forward|excited/gi, weight: -1.2 },
      { regex: /accomplished|achieved|proud/gi, weight: -0.8 },
      { regex: /energetic|motivated|inspired/gi, weight: -0.7 },
      { regex: /connected|supported|loved/gi, weight: -0.9 }
    ]
    
    // Check for positive indicators and adjust score
    positiveIndicators.forEach(indicator => {
      const matches = text.match(indicator.regex) || []
      score += matches.length * indicator.weight
    })
    
    // Ensure score stays within 0-25 range
    return Math.min(Math.max(Math.round(score * 10) / 10, 0), 25)
  }
  
  // Provide interpretation of the depression score
  function getScoreInterpretation(score) {
    if (score < 5) {
      return "Minimal or no depression indicators detected"
    } else if (score < 10) {
      return "Mild depression indicators detected"
    } else if (score < 15) {
      return "Moderate depression indicators detected"
    } else if (score < 20) {
      return "Moderately severe depression indicators detected"
    } else {
      return "Severe depression indicators detected"
    }
  }