'use client'

import React, { useState, useRef } from 'react'
import { Globe, Send } from 'lucide-react'

interface ChatInterfaceProps {
  onSendMessage: (message: string) => void
  disabled?: boolean
  isLoading?: boolean
  useWebSearch?: boolean
  onWebSearchToggle?: (enabled: boolean) => void
}

export default function ChatInterface({
  onSendMessage,
  disabled = false,
  isLoading = false,
  useWebSearch = false,
  onWebSearchToggle,
}: ChatInterfaceProps) {
  const [input, setInput] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || disabled) return

    onSendMessage(input.trim())
    setInput('')
    inputRef.current?.focus()
  }

  const handleWebSearchToggle = () => {
    onWebSearchToggle?.(!useWebSearch)
  }

  return (
    <div className="moving-border-container shadow-xl shadow-blue-100/50">
      <div className={`moving-border-gradient ${isLoading ? 'fast' : ''}`} />
      <div className="moving-border-content px-4 py-3 flex items-center gap-3 bg-white">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Search material pricing, project specs, or history..."
          className="flex-1 bg-transparent border-none focus:ring-0 text-slate-800 placeholder-slate-400 text-sm py-2"
          disabled={disabled}
        />
        <button
          type="button"
          onClick={handleWebSearchToggle}
          className={`p-2 rounded-xl transition-all duration-200 flex items-center gap-2 ${useWebSearch
            ? 'bg-blue-50 text-blue-600 font-medium text-xs'
            : 'text-slate-400 hover:text-slate-600 hover:bg-slate-50'
            }`}
          disabled={disabled}
        >
          <Globe className="w-4 h-4" />
          <span className="hidden sm:inline">Web Search</span>
        </button>
        <button
          type="submit"
          disabled={disabled || !input.trim()}
          className="bg-blue-600 text-white p-2.5 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg shadow-blue-200"
        >
          {isLoading ? (
            <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  )
}
