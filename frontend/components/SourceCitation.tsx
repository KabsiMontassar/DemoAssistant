'use client'

import React from 'react'
import { FileText } from 'lucide-react'

interface SourceCitationProps {
  source: {
    file_path: string
    content_snippet: string
    relevance_score: number
  }
}

export default function SourceCitation({ source, onPreview }: SourceCitationProps & { onPreview?: (source: any) => void }) {
  const scorePercentage = Math.round(source.relevance_score * 100)

  return (
    <div className="bg-gray-50 border border-gray-200 rounded p-2 group hover:border-blue-300 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div
            className="flex items-center gap-2 cursor-pointer hover:underline"
            onClick={() => onPreview?.(source)}
          >
            <FileText size={14} className="text-gray-600 shrink-0" />
            <p className="text-xs font-semibold text-gray-700 truncate" title={source.file_path}>
              {source.file_path}
            </p>
          </div>
          <p className="text-xs text-gray-600 mt-1 line-clamp-2">
            {source.content_snippet}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => onPreview?.(source)}
            className="flex items-center gap-1 px-2 py-1 bg-white border border-slate-200 rounded text-[10px] font-bold text-slate-600 hover:text-blue-600 hover:border-blue-300 transition-all opacity-0 group-hover:opacity-100"
          >
            VIEW
          </button>
          <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-1 rounded">
            {scorePercentage}%
          </span>
        </div>
      </div>
    </div>
  )
}
