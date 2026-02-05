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

export default function SourceCitation({ source }: SourceCitationProps) {
  const scorePercentage = Math.round(source.relevance_score * 100)

  return (
    <div className="bg-gray-50 border border-gray-200 rounded p-2">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <FileText size={14} className="text-gray-600" />
            <p className="text-xs font-semibold text-gray-700">
              {source.file_path}
            </p>
          </div>
          <p className="text-xs text-gray-600 mt-1 line-clamp-2">
            {source.content_snippet}
          </p>
        </div>
        <div className="flex-shrink-0">
          <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-1 rounded">
            {scorePercentage}%
          </span>
        </div>
      </div>
    </div>
  )
}
