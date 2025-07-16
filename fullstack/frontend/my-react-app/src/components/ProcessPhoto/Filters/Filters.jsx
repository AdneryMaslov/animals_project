import { useState } from 'react';
import './filters.css';
import PropTypes from 'prop-types';

const API_URL = import.meta.env.VITE_API_BASE_URL;

export default function Filters({ setParams }) {
    const [coords, setCoords] = useState({ lat: '00.00', lng: '00.00' });
    const [openModal, setOpenModal] = useState(false);
    const [selectedFiles, setSelectedFiles] = useState([]);

    const sendData = async () => {
        if (selectedFiles.length === 0) {
            alert("Пожалуйста, выберите файлы");
            return;
        }

        setParams({ 'results': 'load' });
        setOpenModal(false);

        const formData = new FormData();
        formData.append('cords_sd', coords.lat);
        formData.append('cords_vd', coords.lng);

        for (const file of selectedFiles) {
            formData.append('files', file);
        }

        try {
            const response = await fetch(`${API_URL}/upload/`, {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                throw new Error(`Ошибка сервера: ${response.status}`);
            }
            const res = await response.json();
            setParams({ 'results': 1, 'params': res });
        } catch (error) {
            console.error('Ошибка при загрузке:', error);
            setParams({ 'results': 0 });
            alert(`Не удалось загрузить данные: ${error.message}`);
        }
    };

    const handleFileSelect = (e) => {
        if (e.target.files) {
            setSelectedFiles(Array.from(e.target.files));
        }
    };

    return (
        <>
            <div className="hero-section">
                <div className="hero-overlay"></div>
                <div className="hero-content">
                    <h1>Анализ данных с фотоловушек</h1>
                    <p>Автоматическое распознавание видов животных и отслеживание редких особей с помощью ИИ.</p>
                    <div className="big_add_button" onClick={() => setOpenModal(true)}>
                        НАЧАТЬ РАБОТУ
                    </div>
                </div>
            </div>

            <div className={`modal-upload ${openModal ? 'visible' : 'hidden'}`}>
                <div className="params-container">
                    <div className="close-button" onClick={() => setOpenModal(false)}>
                        &times;
                    </div>
                    <h4>Введите параметры фотоловушки</h4>
                    <div className="params">
                        <div className="param">
                            <div className="input-for">Координаты фотоловушки</div>
                            <div className="input-data">
                                <label>Широта</label>
                                <input type="text" value={coords.lat} onChange={(e) => setCoords({...coords, lat: e.target.value})} />
                            </div>
                            <div className="input-data">
                                <label>Долгота</label>
                                <input type="text" value={coords.lng} onChange={(e) => setCoords({...coords, lng: e.target.value})} />
                            </div>
                        </div>
                    </div>
                    <div className="add-photo">
                        <label className="input-file">
                            <input onChange={handleFileSelect} type="file" name="file" multiple />
                            <span>Выберите файл(ы)</span>
                        </label>
                        <div className="instructions">
                            {selectedFiles.length === 0 &&
                                <p>Загрузите снимок (.jpeg, .png, .jpg) или серию снимков в формате (.zip)</p>
                            }
                            {selectedFiles.length === 1 &&
                                <p>Выбран файл: {selectedFiles[0].name}</p>
                            }
                            {selectedFiles.length > 1 &&
                                <p>Выбрано файлов: {selectedFiles.length}</p>
                            }
                        </div>
                    </div>
                    <button className="send-button" onClick={sendData} disabled={selectedFiles.length === 0}>
                        Отправить на обработку
                    </button>
                </div>
            </div>
        </>
    );
}

Filters.propTypes = {
    setParams: PropTypes.func.isRequired,
};