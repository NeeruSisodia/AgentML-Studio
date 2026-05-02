import { useState, useEffect, useRef } from "react"
import axios from "axios"

const API = "http://localhost:8000"
const WS  = "ws://localhost:8000/ws"

function MetricCard({ label, value, color }) {
  return (
    <div style={{
      background: "#f8faff",
      borderRadius: 10,
      padding: "14px 16px",
      textAlign: "center",
      flex: 1,
      minWidth: 120
    }}>
      <div style={{
        fontSize: 11,
        color: "#888",
        marginBottom: 4,
        textTransform: "uppercase",
        letterSpacing: ".05em"
      }}>
        {label}
      </div>
      <div style={{
        fontSize: 20,
        fontWeight: 600,
        color: color || "#1a1a2e"
      }}>
        {value}
      </div>
    </div>
  )
}

export default function App() {
  const [stage,       setStage]       = useState("idle")
  const [logs,        setLogs]        = useState([])
  const [met,         setMet]         = useState({})
  const [dec,         setDec]         = useState("—")
  const [conf,        setConf]        = useState(0)
  const [status,      setStatus]      = useState("Upload a CSV file to begin")
  const [activeTab,   setActiveTab]   = useState("ml")
  const [fileResult,  setFileResult]  = useState(null)
  const [fileLoading, setFileLoading] = useState(false)
  const [customQ,     setCustomQ]     = useState("")

  const wsRef  = useRef(null)
  const logRef = useRef(null)

  useEffect(() => {
    wsRef.current = new WebSocket(WS)
    wsRef.current.onmessage = (e) => {
      const d = JSON.parse(e.data)
      setStage(d.stage)
      setLogs(d.logs || [])
      setMet(d.metrics || {})
      setDec(d.decision || "—")
      setConf(d.confidence || 0)
      if (d.stage === "complete") {
        setStatus("Pipeline complete!")
      } else if (d.stage !== "idle") {
        setStatus(`Running: ${d.stage}...`)
      }
    }
    wsRef.current.onerror = () => {
      setStatus("Connection error — is backend running?")
    }
    //  Reconnect if connection is lost
    wsRef.current.onclose = () => {
      setStatus("Connection lost — retrying...")
      setTimeout(() => {
        wsRef.current = new WebSocket(WS)
      }, 3000)
    }
    return () => wsRef.current?.close()
  }, [])

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setStatus(`Uploading ${file.name}...`)
    setLogs([])
    setMet({})
    setDec("—")
    setConf(0)
    const form = new FormData()
    form.append("file", file)
    try {
      await axios.post(`${API}/api/run`, form)
      setStatus("Agent is running pipeline...")
    } catch {
      setStatus("Error: make sure backend is running")
    }
  }

  const handleAnyFile = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setFileLoading(true)
    setFileResult(null)
    setStatus(`Analysing ${file.name}...`)
    const form = new FormData()
    form.append("file", file)
    try {
      const url = customQ
        ? `${API}/api/analyse-file?question=${encodeURIComponent(customQ)}`
        : `${API}/api/analyse-file`
      const res = await axios.post(url, form)
      setFileResult(res.data)
      setStatus("Analysis complete!")
    } catch {
      setFileResult({
        status: "error",
        error: "Analysis failed. Check backend is running."
      })
      setStatus("Analysis failed")
    } finally {
      setFileLoading(false)
    }
  }

  const stages = [
    "starting", "ingestion", "analysis",
    "features", "training", "deployment", "complete"
  ]

  return (
    <div style={{
      maxWidth: 860,
      margin: "0 auto",
      padding: "24px 16px",
      fontFamily: "system-ui, sans-serif"
    }}>

      {/* Header */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        marginBottom: 24
      }}>
        <div style={{
          width: 44,
          height: 44,
          borderRadius: 10,
          background: "#4F46E5",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#fff",
          fontWeight: 700,
          fontSize: 20
        }}>
          A
        </div>
        <div>
          <div style={{
            fontSize: 20,
            fontWeight: 700,
            color: "#1a1a2e"
          }}>
            AgentML Studio
          </div>
          <div style={{ fontSize: 13, color: "#888" }}>
            Agentic AI — LangChain + TinyLlama FREE
          </div>
        </div>
        <div style={{
          marginLeft: "auto",
          display: "flex",
          alignItems: "center",
          gap: 6
        }}>
          <div style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: stage === "complete"
              ? "#10b981" : "#3b82f6"
          }} />
          <span style={{ fontSize: 12, color: "#888" }}>
            {status}
          </span>
        </div>
      </div>

      {/* Tab Buttons */}
      <div style={{
        display: "flex",
        gap: 8,
        marginBottom: 20
      }}>
        <button
          onClick={() => setActiveTab("ml")}
          style={{
            padding: "10px 24px",
            borderRadius: 8,
            border: "none",
            cursor: "pointer",
            fontSize: 13,
            fontWeight: 600,
            background: activeTab === "ml"
              ? "#4F46E5" : "#f1f5f9",
            color: activeTab === "ml"
              ? "#fff" : "#64748b",
            transition: "all .2s"
          }}
        >
          ML Pipeline
        </button>
        <button
          onClick={() => {
            setActiveTab("files")
            setFileResult(null)
          }}
          style={{
            padding: "10px 24px",
            borderRadius: 8,
            border: "none",
            cursor: "pointer",
            fontSize: 13,
            fontWeight: 600,
            background: activeTab === "files"
              ? "#7C3AED" : "#f1f5f9",
            color: activeTab === "files"
              ? "#fff" : "#64748b",
            transition: "all .2s"
          }}
        >
          File Analyser
        </button>
      </div>

      {/* ── ML PIPELINE TAB ────────────────────── */}
      {activeTab === "ml" && (
        <div>

          {/* Upload CSV */}
          <div style={{
            border: "2px dashed #e2e8f0",
            borderRadius: 12,
            padding: 24,
            textAlign: "center",
            marginBottom: 20,
            background: "#fafbff"
          }}>
            <div style={{
              fontSize: 14,
              color: "#666",
              marginBottom: 12
            }}>
              Drop a CSV file or click to upload
            </div>
            <input
              type="file"
              accept=".csv"
              onChange={handleUpload}
              style={{ fontSize: 13, cursor: "pointer" }}
            />
            <div style={{
              fontSize: 12,
              color: "#aaa",
              marginTop: 8
            }}>
              Try titanic.csv or any dataset
            </div>
          </div>

          {/* Stage pills */}
          <div style={{
            display: "flex",
            gap: 6,
            marginBottom: 20,
            flexWrap: "wrap"
          }}>
            {stages.map(s => (
              <div key={s} style={{
                padding: "5px 12px",
                borderRadius: 20,
                fontSize: 12,
                fontWeight: 500,
                background: stage === s
                  ? "#4F46E5"
                  : stage === "complete"
                    ? "#d1fae5" : "#f1f5f9",
                color: stage === s
                  ? "#fff"
                  : stage === "complete"
                    ? "#065f46" : "#64748b",
                border: `1px solid ${
                  stage === s ? "#4F46E5"
                  : stage === "complete"
                    ? "#a7f3d0" : "#e2e8f0"
                }`
              }}>
                {s}
              </div>
            ))}
          </div>

          {/* Metrics */}
          <div style={{
            display: "flex",
            gap: 10,
            marginBottom: 20,
            flexWrap: "wrap"
          }}>
            <MetricCard
              label="Rows"
              value={met.rows || "—"}
            />
            <MetricCard
              label="Features"
              value={met.n_features || "—"}
            />
            <MetricCard
              label="Accuracy"
              value={met.accuracy
                ? (met.accuracy * 100).toFixed(1) + "%"
                : "—"}
              color="#4F46E5"
            />
            <MetricCard
              label="F1 Score"
              value={met.f1
                ? met.f1.toFixed(3) : "—"}
            />
            <MetricCard
              label="Best Model"
              value={met.model_name || "—"}
            />
            <MetricCard
              label="Decision"
              value={dec}
              color={
                dec === "AUTO DEPLOYED"
                  ? "#10b981"
                  : dec === "NEEDS HUMAN REVIEW"
                    ? "#f59e0b" : "#ef4444"
              }
            />
          </div>

          {/* Confidence bar */}
          {conf > 0 && (
            <div style={{ marginBottom: 20 }}>
              <div style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 13,
                marginBottom: 5
              }}>
                <span style={{ color: "#666" }}>
                  Agent confidence
                </span>
                <span style={{ fontWeight: 600 }}>
                  {(conf * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{
                height: 8,
                background: "#e2e8f0",
                borderRadius: 4,
                overflow: "hidden"
              }}>
                <div style={{
                  width: `${conf * 100}%`,
                  height: "100%",
                  borderRadius: 4,
                  transition: "width .8s ease",
                  background: conf > 0.82
                    ? "#10b981"
                    : conf > 0.65
                      ? "#f59e0b" : "#ef4444"
                }} />
              </div>
            </div>
          )}

          {/* Feature importance */}
          {met.shap && (
            <div style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 12,
              padding: 16,
              marginBottom: 20
            }}>
              <div style={{
                fontSize: 14,
                fontWeight: 600,
                marginBottom: 12
              }}>
                Feature importance
              </div>
              {Object.entries(met.shap)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 6)
                .map(([k, v]) => (
                  <div key={k} style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 8
                  }}>
                    <span style={{
                      fontSize: 12,
                      color: "#555",
                      width: 140,
                      flexShrink: 0,
                      textAlign: "right"
                    }}>
                      {k.length > 16
                        ? k.slice(0, 15) + "…" : k}
                    </span>
                    <div style={{
                      flex: 1,
                      height: 8,
                      background: "#f1f5f9",
                      borderRadius: 4,
                      overflow: "hidden"
                    }}>
                      <div style={{
                        width: `${Math.min(
                          v / Math.max(
                            ...Object.values(met.shap)
                          ) * 100, 100
                        )}%`,
                        height: "100%",
                        background: "#4F46E5",
                        borderRadius: 4,
                        transition: "width 1s ease"
                      }} />
                    </div>
                    <span style={{
                      fontSize: 11,
                      color: "#888",
                      width: 40,
                      textAlign: "right"
                    }}>
                      {v.toFixed(3)}
                    </span>
                  </div>
                ))}
            </div>
          )}

          {/* Log terminal */}
          <div
            ref={logRef}
            style={{
              background: "#0f172a",
              borderRadius: 10,
              padding: 16,
              height: 220,
              overflowY: "auto",
              fontFamily: "monospace",
              fontSize: 12.5
            }}
          >
            {logs.length === 0 && (
              <span style={{ color: "#4a5568" }}>
                Waiting for agent to start...
              </span>
            )}
            {logs.map((l, i) => (
              <div key={i} style={{
                marginBottom: 4,
                color: l.level === "error"
                  ? "#f87171"
                  : l.level === "warn"
                    ? "#fbbf24" : "#86efac"
              }}>
                [{l.time}] {l.msg}
              </div>
            ))}
          </div>

        </div>
      )}

      {/* ── FILE ANALYSER TAB ──────────────────── */}
      {activeTab === "files" && (
        <div>

          {/* Upload any file */}
          <div style={{
            border: "2px dashed #e2e8f0",
            borderRadius: 12,
            padding: 24,
            textAlign: "center",
            marginBottom: 16,
            background: "#faf5ff"
          }}>
            <div style={{
              fontSize: 15,
              fontWeight: 600,
              color: "#7C3AED",
              marginBottom: 8
            }}>
              Upload Any File for AI Analysis
            </div>
            <div style={{
              fontSize: 12,
              color: "#888",
              marginBottom: 16
            }}>
              CSV · Excel · Word · PDF · Images · Text
            </div>
            <input
              type="file"
              accept=".csv,.xlsx,.xls,.docx,.doc,.pdf,.txt,.md,.jpg,.jpeg,.png,.gif,.bmp,.webp"
              onChange={handleAnyFile}
              style={{
                fontSize: 13,
                cursor: "pointer",
                display: "block",
                margin: "0 auto 12px"
              }}
            />
            <input
              type="text"
              placeholder="Ask a specific question (optional)"
              value={customQ}
              onChange={e => setCustomQ(e.target.value)}
              style={{
                width: "80%",
                padding: "8px 12px",
                borderRadius: 8,
                border: "1px solid #e2e8f0",
                fontSize: 13,
                marginTop: 8
              }}
            />
            <div style={{
              fontSize: 11,
              color: "#aaa",
              marginTop: 8
            }}>
              For images you can ask specific questions
            </div>
          </div>

          {/* Loading */}
          {fileLoading && (
            <div style={{
              textAlign: "center",
              padding: 24,
              fontSize: 14,
              color: "#7C3AED"
            }}>
              AI is analysing your file...
              Please wait...
            </div>
          )}

          {/* Result */}
          {fileResult && !fileLoading && (
            <div style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 12,
              padding: 20
            }}>

              {/* Result header */}
              <div style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 16
              }}>
                <div style={{
                  fontSize: 15,
                  fontWeight: 600
                }}>
                  Analysis Result
                </div>
                <div style={{
                  display: "flex",
                  gap: 8,
                  alignItems: "center"
                }}>
                  {fileResult.file_name && (
                    <span style={{
                      background: "#f1f5f9",
                      padding: "3px 10px",
                      borderRadius: 20,
                      fontSize: 11,
                      color: "#555"
                    }}>
                      {fileResult.file_name}
                    </span>
                  )}
                  {fileResult.type && (
                    <span style={{
                      background: fileResult.status === "success"
                        ? "#d1fae5" : "#fee2e2",
                      color: fileResult.status === "success"
                        ? "#065f46" : "#991b1b",
                      padding: "3px 10px",
                      borderRadius: 20,
                      fontSize: 11,
                      fontWeight: 600
                    }}>
                      {fileResult.type.toUpperCase()}
                    </span>
                  )}
                </div>
              </div>

              {/* Error message */}
              {fileResult.status === "error" && (
                <div style={{
                  color: "#ef4444",
                  fontSize: 13,
                  padding: 12,
                  background: "#fef2f2",
                  borderRadius: 8,
                  marginBottom: 12
                }}>
                  Error: {fileResult.error}
                </div>
              )}

              {/* Summary for PDF Word Text */}
              {fileResult.summary && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: "#7C3AED",
                    marginBottom: 6,
                    textTransform: "uppercase",
                    letterSpacing: ".05em"
                  }}>
                    AI Summary
                  </div>
                  <div style={{
                    fontSize: 13,
                    color: "#444",
                    lineHeight: 1.8,
                    padding: 12,
                    background: "#faf5ff",
                    borderRadius: 8
                  }}>
                    {fileResult.summary}
                  </div>
                </div>
              )}

              {/* Image description */}
              {fileResult.description && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: "#7C3AED",
                    marginBottom: 6,
                    textTransform: "uppercase",
                    letterSpacing: ".05em"
                  }}>
                    Image Analysis
                  </div>
                  <div style={{
                    fontSize: 13,
                    color: "#444",
                    lineHeight: 1.8,
                    padding: 12,
                    background: "#faf5ff",
                    borderRadius: 8
                  }}>
                    {fileResult.description}
                  </div>
                </div>
              )}

              {/* CSV columns */}
              {fileResult.column_names && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: "#7C3AED",
                    marginBottom: 8,
                    textTransform: "uppercase",
                    letterSpacing: ".05em"
                  }}>
                    Columns Found
                  </div>
                  <div style={{
                    display: "flex",
                    gap: 6,
                    flexWrap: "wrap"
                  }}>
                    {fileResult.column_names.map(col => (
                      <span key={col} style={{
                        background: "#f1f5f9",
                        padding: "3px 10px",
                        borderRadius: 20,
                        fontSize: 12,
                        color: "#555"
                      }}>
                        {col}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Excel sheets */}
              {fileResult.sheets && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: "#7C3AED",
                    marginBottom: 8,
                    textTransform: "uppercase",
                    letterSpacing: ".05em"
                  }}>
                    Excel Sheets
                  </div>
                  {Object.entries(
                    fileResult.sheets
                  ).map(([name, info]) => (
                    <div key={name} style={{
                      background: "#faf5ff",
                      borderRadius: 8,
                      padding: 12,
                      marginBottom: 8
                    }}>
                      <div style={{
                        fontWeight: 600,
                        fontSize: 13,
                        marginBottom: 4
                      }}>
                        Sheet: {name}
                      </div>
                      <div style={{
                        fontSize: 12,
                        color: "#666"
                      }}>
                        {info.rows} rows,
                        {" "}{info.columns} columns
                      </div>
                      <div style={{
                        fontSize: 11,
                        color: "#888",
                        marginTop: 4
                      }}>
                        {info.column_names
                          ?.slice(0, 5)
                          .join(", ")}
                        {info.column_names?.length > 5
                          ? "..." : ""}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Stats row */}
              <div style={{
                display: "flex",
                gap: 8,
                flexWrap: "wrap",
                paddingTop: 12,
                borderTop: "1px solid #f1f5f9"
              }}>
                {fileResult.rows !== undefined && (
                  <span style={{
                    background: "#f1f5f9",
                    padding: "4px 12px",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "#555"
                  }}>
                    Rows: {fileResult.rows}
                  </span>
                )}
                {fileResult.pages && (
                  <span style={{
                    background: "#f1f5f9",
                    padding: "4px 12px",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "#555"
                  }}>
                    Pages: {fileResult.pages}
                  </span>
                )}
                {fileResult.word_count && (
                  <span style={{
                    background: "#f1f5f9",
                    padding: "4px 12px",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "#555"
                  }}>
                    Words: {fileResult.word_count}
                  </span>
                )}
                {fileResult.paragraphs && (
                  <span style={{
                    background: "#f1f5f9",
                    padding: "4px 12px",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "#555"
                  }}>
                    Paragraphs: {fileResult.paragraphs}
                  </span>
                )}
                {fileResult.tables !== undefined && (
                  <span style={{
                    background: "#f1f5f9",
                    padding: "4px 12px",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "#555"
                  }}>
                    Tables: {fileResult.tables}
                  </span>
                )}
                {fileResult.file_size_kb && (
                  <span style={{
                    background: "#f1f5f9",
                    padding: "4px 12px",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "#555"
                  }}>
                    Size: {fileResult.file_size_kb} KB
                  </span>
                )}
                {fileResult.metadata?.width && (
                  <span style={{
                    background: "#f1f5f9",
                    padding: "4px 12px",
                    borderRadius: 8,
                    fontSize: 12,
                    color: "#555"
                  }}>
                    {fileResult.metadata.width}x
                    {fileResult.metadata.height}px
                    {" "}{fileResult.metadata.format}
                  </span>
                )}
              </div>

            </div>
          )}

        </div>
      )}

    </div>
  )
}
