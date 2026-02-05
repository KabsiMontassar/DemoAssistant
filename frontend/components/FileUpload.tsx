'use client'

import React, { useRef, useState } from 'react'
import { Upload } from 'lucide-react'

interface FileUploadProps {
  onFileSelect: (file: File) => void
  disabled?: boolean
}

export default function FileUpload({ onFileSelect, disabled = false }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files.length > 0) {
      const file = files[0]
      if (file.name.endsWith('.txt')) {
        onFileSelect(file)
      } else {
        alert('Only .txt files are supported')
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (files && files.length > 0) {
      onFileSelect(files[0])
    }
  }

  return (
    <div className="relative">
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt"
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled}
      />

      <div
        onClick={() => !disabled && fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`px-4 py-2 border-2 rounded-lg cursor-pointer transition-colors ${
          disabled
            ? 'border-gray-300 bg-gray-100 text-gray-500 cursor-not-allowed'
            : isDragging
            ? 'border-blue-500 bg-blue-50 text-blue-600'
            : 'border-gray-300 hover:border-blue-500 text-gray-700'
        }`}
      >
        <div className="flex items-center gap-2">
          <Upload size={16} />
          <span className="text-sm font-medium">
            {isDragging ? 'Drop file here' : 'Upload File'}
          </span>
        </div>
      </div>
    </div>
  )
}
