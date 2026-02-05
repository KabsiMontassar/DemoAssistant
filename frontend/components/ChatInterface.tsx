'use client'

import React, { useState, useRef } from 'react'
import { Globe } from 'lucide-react'

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
    <form onSubmit={handleSubmit} className="relative flex gap-2">
      <div className="flex-1 relative">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about material pricing, projects, or specifications..."
          disabled={disabled}
          className={`w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
            disabled ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'
          }`}
        />
        <button
          type="button"
          onClick={handleWebSearchToggle}
          disabled={disabled}
          title={`Web search: ${useWebSearch ? 'enabled' : 'disabled'}`}
          className={`absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded transition-colors ${
            useWebSearch
              ? 'text-blue-500 hover:text-blue-600'
              : 'text-gray-400 hover:text-gray-500'
          } ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
        >
          <Globe size={20} />
        </button>
      </div>
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
