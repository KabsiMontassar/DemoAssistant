'use client'

import React, { useState, useRef } from 'react'

interface ChatInterfaceProps {
  onSendMessage: (message: string) => void
  disabled?: boolean
  isLoading?: boolean
}

export default function ChatInterface({
  onSendMessage,
  disabled = false,
  isLoading = false,
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

  return (
    <form onSubmit={handleSubmit} className="flex gap-3">
      <input
        ref={inputRef}
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask about material pricing, projects, or specifications..."
        disabled={disabled}
        className={`flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
          disabled ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'
        }`}
      />
      <button
        type="submit"
        disabled={disabled || !input.trim()}
        className={`px-6 py-3 rounded-lg font-medium transition-colors ${
          disabled || !input.trim()
            ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
            : 'bg-blue-500 text-white hover:bg-blue-600 active:bg-blue-700'
        }`}
      >
        {isLoading ? (
          <span className="flex items-center gap-2">
            <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Sending...
          </span>
        ) : (
          'Send'
        )}
      </button>
    </form>
  )
}
