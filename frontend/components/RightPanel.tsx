'use client'

import React from 'react'
import { X, FileText, Download, Table as TableIcon } from 'lucide-react'
import * as XLSX from 'xlsx'

interface RightPanelProps {
    isOpen: boolean
    onClose: () => void
    file: {
        path: string
        content?: string
        url?: string
        type: 'pdf' | 'csv' | 'excel' | 'text' | 'image' | 'unknown'
    } | null
}

const TableViewer = ({ headers, rows }: { headers: string[], rows: any[][] }) => {
    return (
        <div className="w-full h-full overflow-auto bg-white p-4">
            <table className="min-w-full divide-y divide-gray-200 border">
                <thead className="bg-gray-50 sticky top-0">
                    <tr>
                        {headers.map((header, i) => (
                            <th key={i} className="px-4 py-2 text-left text-xs font-bold text-gray-500 uppercase tracking-wider border">
                                {header || `Col ${i + 1}`}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-100">
                    {rows.map((row, i) => (
                        <tr key={i} className="hover:bg-blue-50/10 transition-colors">
                            {row.map((cell, j) => (
                                <td key={j} className="px-4 py-2 text-sm text-gray-600 border whitespace-nowrap min-w-[100px]">
                                    {cell?.toString() || ''}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    )
}

const SheetViewer = ({ url }: { url: string }) => {
    const [data, setData] = React.useState<{ headers: string[], rows: any[][] } | null>(null)
    const [loading, setLoading] = React.useState(true)
    const [error, setError] = React.useState<string | null>(null)

    React.useEffect(() => {
        const fetchFile = async () => {
            try {
                const response = await fetch(url)

                if (url.toLowerCase().endsWith('.csv')) {
                    const text = await response.text()
                    const workbook = XLSX.read(text, { type: 'string' })
                    const firstSheetName = workbook.SheetNames[0]
                    const worksheet = workbook.Sheets[firstSheetName]
                    const jsonData: any[][] = XLSX.utils.sheet_to_json(worksheet, { header: 1 })

                    if (jsonData.length > 0) {
                        setData({
                            headers: jsonData[0].map((h: any) => h?.toString() || ''),
                            rows: jsonData.slice(1)
                        })
                    }
                } else {
                    const arrayBuffer = await response.arrayBuffer()
                    const workbook = XLSX.read(arrayBuffer, { type: 'array' })
                    const firstSheetName = workbook.SheetNames[0]
                    const worksheet = workbook.Sheets[firstSheetName]
                    const jsonData: any[][] = XLSX.utils.sheet_to_json(worksheet, { header: 1 })

                    if (jsonData.length > 0) {
                        setData({
                            headers: jsonData[0].map((h: any) => h?.toString() || ''),
                            rows: jsonData.slice(1)
                        })
                    }
                }
                setLoading(false)
            } catch (err) {
                console.error('Error parsing sheet:', err)
                setError('Failed to parse file content')
                setLoading(false)
            }
        }
        fetchFile()
    }, [url])

    if (loading) return <div className="p-10 text-center text-slate-400">Loading spreadsheet...</div>
    if (error) return <div className="p-10 text-center text-red-400">{error}</div>
    if (!data) return <div className="p-10 text-center text-slate-400">No data found</div>

    return <TableViewer headers={data.headers} rows={data.rows} />
}

const TextViewer = ({ url, initialContent }: { url: string, initialContent?: string }) => {
    const [content, setContent] = React.useState<string | null>(initialContent || null)
    const [loading, setLoading] = React.useState(!initialContent)

    React.useEffect(() => {
        if (!initialContent && url) {
            fetch(url)
                .then(res => res.text())
                .then(text => {
                    setContent(text)
                    setLoading(false)
                })
                .catch(() => setLoading(false))
        }
    }, [url, initialContent])

    if (loading) return <div className="p-10 text-center text-slate-400">Loading file content...</div>

    return (
        <div className="p-6 h-full overflow-y-auto">
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
                <pre className="text-sm text-slate-700 font-mono whitespace-pre-wrap break-words">
                    {content || "No content found"}
                </pre>
            </div>
        </div>
    )
}

export default function RightPanel({ isOpen, onClose, file }: RightPanelProps) {
    if (!isOpen || !file) return null

    return (
        <div className="w-[650px] border-l border-gray-200 bg-white shadow-sm flex flex-col h-full animate-in slide-in-from-right duration-300 z-30">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 bg-white shrink-0">
                <div className="flex items-center gap-3 overflow-hidden">
                    <div className="p-2 bg-blue-50 rounded-lg shrink-0">
                        {file.type === 'csv' || file.type === 'excel' ? <TableIcon size={20} className="text-blue-600" /> : <FileText size={20} className="text-blue-600" />}
                    </div>
                    <div className="flex flex-col min-w-0">
                        <h2 className="text-sm font-bold text-slate-800 truncate" title={file.path}>
                            {file.path.split('/').pop()}
                        </h2>
                        <p className="text-xs text-slate-500 truncate" title={file.path}>
                            {file.path}
                        </p>
                    </div>
                </div>
                <button
                    onClick={onClose}
                    className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-full transition-colors"
                >
                    <X size={20} />
                </button>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-hidden bg-slate-50 relative">
                {file.type === 'pdf' ? (
                    <iframe
                        src={file.url}
                        className="w-full h-full border-none bg-white"
                        title="PDF Preview"
                    />
                ) : file.type === 'image' ? (
                    <div className="w-full h-full flex items-center justify-center p-8 bg-slate-200/30">
                        <img src={file.url} alt="Preview" className="max-w-full max-h-full shadow-lg rounded" />
                    </div>
                ) : (file.type === 'csv' || file.type === 'excel') ? (
                    <SheetViewer url={file.url!} />
                ) : (
                    <TextViewer url={file.url!} initialContent={file.content} />
                )}
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-gray-100 bg-white shrink-0">
                <button
                    className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-slate-100 text-slate-700 font-medium rounded-lg hover:bg-slate-200 transition-colors"
                    onClick={() => {
                        window.location.href = file.url?.replace('/api/view', '/api/download') || '';
                    }}
                >
                    <Download size={16} />
                    Download File
                </button>
            </div>
        </div>
    )
}
