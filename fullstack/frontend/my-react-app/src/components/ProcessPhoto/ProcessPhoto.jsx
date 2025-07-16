import Filters from './Filters/Filters'
import Result from './Result/Result'
import './ProcessPhoto.css'
import { useState } from 'react'

export default function ProcessPhoto() {
    const [params, setParams] = useState({"results": 0})
   
    if (params.results === 0) {
        return <Filters setParams={setParams} />
    } else if (params.results === 'load') {
        return (
            <div className='loading-overlay'>
                <div className='loader'></div>
                <div className='loadText'>Подождите, идет загрузка и обработка...</div>
            </div>
        )
    } else {
        return <Result params={params} setParams={setParams} />
    }
}
