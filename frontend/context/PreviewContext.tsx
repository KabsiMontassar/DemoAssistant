'use client'

import React, { createContext, useContext, useState, ReactNode } from 'react'
import { getApiUrl } from '@/lib/api'

interface PreviewFile {
    path: string
    content?: string  // For text snippets from chat
    url?: string      // For full file viewing (PDF/Binary)
    type: 'pdf' | 'csv' | 'excel' | 'text' | 'image' | 'unknown'
}

interface PreviewContextType {
    isOpen: boolean
    selectedFile: PreviewFile | null
    openPreview: (path: string, content?: string) => void
    closePreview: () => void
}

const PreviewContext = createContext<PreviewContextType | undefined>(undefined)

export function PreviewProvider({ children }: { children: ReactNode }) {
    const [isOpen, setIsOpen] = useState(false)
    const [selectedFile, setSelectedFile] = useState<PreviewFile | null>(null)

    const openPreview = (path: string, content?: string) => {
        const apiBaseUrl = getApiUrl()
        const ext = path.split('.').pop()?.toLowerCase() || ''

        let type: PreviewFile['type'] = 'unknown'
        if (ext === 'pdf') type = 'pdf'
        else if (ext === 'csv') type = 'csv'
        else if (ext === 'xlsx' || ext === 'xls') type = 'excel'
        else if (['png', 'jpg', 'jpeg', 'gif', 'svg'].includes(ext)) type = 'image'
        else if (['txt', 'md', 'py', 'js', 'json', 'csv'].includes(ext)) type = 'text'

        const url = `${apiBaseUrl}/api/view?path=${encodeURIComponent(path)}`

        setSelectedFile({
            path,
            content,
            url,
            type
        })
        setIsOpen(true)
    }

    const closePreview = () => {
        setIsOpen(false)
    }

    return (
        <PreviewContext.Provider value={{ isOpen, selectedFile, openPreview, closePreview }}>
            {children}
        </PreviewContext.Provider>
    )
}

export function usePreview() {
    const context = useContext(PreviewContext)
    if (context === undefined) {
        throw new Error('usePreview must be used within a PreviewProvider')
    }
    return context
}
