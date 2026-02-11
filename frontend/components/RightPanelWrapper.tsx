'use client'

import { usePreview } from '@/context/PreviewContext'
import RightPanel from './RightPanel'

export default function RightPanelWrapper() {
    const { isOpen, closePreview, selectedFile } = usePreview()

    return (
        <RightPanel
            isOpen={isOpen}
            onClose={closePreview}
            file={selectedFile}
        />
    )
}
