import React, { useState } from 'react'
import './styles.css'

function App(){
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [transcript, setTranscript] = useState('')

  async function sendQuery(){
    setAnswer('Thinking...')
    const resp = await fetch('/query', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({query})
    })
    const data = await resp.json()
    setAnswer(data.answer)
    setSources(data.sources || [])
  }

  async function uploadAudio(file){
    const fd = new FormData()
    fd.append('file', file)
    const r = await fetch('/transcribe', {method:'POST', body:fd})
    const d = await r.json()
    setTranscript(d.text)
    setQuery(d.text)
  }

  return (
    <div className="container">
      <h1>Ultron</h1>
      <div className="panel">
        <textarea value={query} onChange={e=>setQuery(e.target.value)} placeholder="Ask about astrophysics, rockets, nuclear physics..." />
        <div className="controls">
          <button onClick={sendQuery}>Ask</button>
          <label className="upload-btn">
            Upload audio
            <input type="file" accept="audio/*" onChange={e=>uploadAudio(e.target.files[0])} />
          </label>
        </div>
      </div>
      <div className="result">
        <h2>Answer</h2>
        <pre>{answer}</pre>
        <h3>Sources</h3>
        <ul>{sources.map((s,i)=>(<li key={i}><pre>{s.preview}</pre></li>))}</ul>
      </div>
    </div>
  )
}

export default App
