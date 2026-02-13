'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Globe, Send, Search, Sparkles } from 'lucide-react'

interface ChatInterfaceProps {
  onSendMessage: (message: string) => void
  disabled?: boolean
  isLoading?: boolean
  currentStatus?: string | null
  useWebSearch?: boolean
  onWebSearchToggle?: (enabled: boolean) => void
}

export default function ChatInterface({
  onSendMessage,
  disabled = false,
  isLoading = false,
  currentStatus = null,
  useWebSearch = false,
  onWebSearchToggle,
}: ChatInterfaceProps) {
  const [input, setInput] = useState('')
  const [isFocused, setIsFocused] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = (e?: React.FormEvent | React.KeyboardEvent) => {
    if (e) e.preventDefault()
    if (!input.trim() || disabled || isLoading) return

    onSendMessage(input.trim())
    setInput('')

    // Auto-resize back to base row
    if (inputRef.current) {
      inputRef.current.style.height = 'auto'
    }

    inputRef.current?.focus()
  }

  const handleWebSearchToggle = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    onWebSearchToggle?.(!useWebSearch)
    inputRef.current?.focus()
  }

  // Auto-focus on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  return (
    <div className="w-full pb-8 pt-1 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Main Input Container */}
        <form
          onSubmit={handleSubmit}
          className="relative"
        >
          {/* Animated Border Container */}
          <div className={`relative rounded-xl transition-all duration-500 ${isLoading
            ? 'bg-gradient-to-r from-blue-300 to-blue-400 p-[1.5px] animate-gradient-x'
            : ' bg-gradient-to-r from-blue-100 to-blue-200 p-[1px]'
            }`}>
            {/* Inner Content */}
            <div className={`relative bg-white rounded-xl overflow-hidden transition-all duration-300 ${isFocused ? 'shadow-lg shadow-blue-100/50' : ''}`}>
              {/* Input Area */}
              <div className="flex items-start p-1 min-h-[56px]">

                {/* Textarea for Multi-line Support */}
                <div className="flex-1 relative">
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        handleSubmit(e)
                      }
                    }}
                    onFocus={() => setIsFocused(true)}
                    onBlur={() => setIsFocused(false)}
                    placeholder="Ask about material properties, applications..."
                    className="w-full px-4 py-4 bg-transparent border-none focus:ring-0 text-slate-800 placeholder-slate-400/70 text-[15px] font-normal leading-relaxed outline-none resize-none overflow-hidden max-h-32"
                    disabled={disabled || isLoading}
                    rows={1}
                    onInput={(e) => {
                      const target = e.target as HTMLTextAreaElement
                      target.style.height = 'auto'
                      target.style.height = `${Math.min(target.scrollHeight, 128)}px`
                    }}
                  />

                  {/* Loading Indicator */}
                  {isLoading && (
                    <div className="absolute right-4 top-1/2 transform -translate-y-1/2 flex items-center gap-3 bg-white/80 backdrop-blur-sm pl-2">
                      <div className="flex items-center gap-1">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="flex items-center gap-1 pr-2 pt-2">
                  {/* Web Search Toggle */}
                  {onWebSearchToggle && (
                    <button
                      type="button"
                      onClick={handleWebSearchToggle}
                      disabled={disabled || isLoading}
                      className={`p-2.5 rounded-xl transition-all duration-300 ${useWebSearch
                        ? 'bg-blue-600 text-white shadow-md'
                        : 'text-slate-400 hover:text-blue-600 hover:bg-blue-50'
                        } ${disabled || isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105 active:scale-95'}`}
                      title={useWebSearch ? "Disable web search" : "Enable web search"}
                    >
                      <Globe size={18} strokeWidth={useWebSearch ? 2.5 : 2} />
                    </button>
                  )}

                  {/* Send Button */}
                  <button
                    type="submit"
                    disabled={disabled || !input.trim() || isLoading}
                    className={`p-2.5 rounded-xl flex items-center justify-center transition-all duration-300 ${disabled || !input.trim() || isLoading
                      ? 'bg-slate-100 text-slate-300 cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:scale-105 active:scale-95'
                      }`}
                  >
                    <Send size={18} strokeWidth={2.5} />
                  </button>
                </div>
              </div>

              {/* Bottom Instructions Bar */}
              <div className="px-4 pb-2">
                <div className="h-px bg-gradient-to-r from-transparent via-blue-100/50 to-transparent mb-2"></div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <span className={`text-[10px] font-bold tracking-tight transition-colors duration-300 ${input.length > 400 ? 'text-amber-600' : 'text-slate-400'}`}>
                      {input.length}/500
                    </span>
                    <span className="text-[10px] font-medium text-slate-400 uppercase tracking-widest hidden sm:inline">
                      Enter to Send • Shift+Enter for New Line
                    </span>
                  </div>

                </div>
              </div>
            </div>
          </div>
        </form>
      </div>

      <style jsx>{`
        @keyframes gradient-x {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .animate-gradient-x {
          background-size: 200% 200%;
          animation: gradient-x 3s ease infinite;
        }
      `}</style>
    </div>
  )
}