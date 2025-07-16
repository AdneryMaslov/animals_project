import { useState, useCallback, useEffect } from 'react';
import { PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';
import './result.css';

const API_URL = import.meta.env.VITE_API_BASE_URL;

const RARE_ANIMALS_LIST = ["Зубр", "Выдра", "Рысь", "Норка"];

const ActionableRow = ({ item, allPassports, onActionComplete, onOpenAddModal, coordinates, onOpenImageModal }) => {
    const [isActionPanelOpen, setIsActionPanelOpen] = useState(false);
    const [selectedPassportId, setSelectedPassportId] = useState('');

    const relevantPassports = allPassports.filter(p => p.type === item.type);

    const handleAssign = async () => {
        if (!selectedPassportId) {
            onActionComplete('Пожалуйста, выберите паспорт.', 'error');
            return;
        }

        const originalCoords = coordinates || '0.0,0.0';
        const [lat, lng] = originalCoords.split(',');

        const formData = new FormData();
        formData.append('image_name', item.IMG);
        formData.append('passport_id', selectedPassportId);
        formData.append('cords_sd', lat.trim());
        formData.append('cords_vd', lng.trim());

        try {
            const response = await fetch(`${API_URL}/assign_passport/`, { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Не удалось присвоить паспорт.');

            const result = await response.json();

            onActionComplete('Фото успешно присвоено!', 'success', {
                img: item.IMG,
                passportId: result.passport_id
            });
            setIsActionPanelOpen(false);
        } catch (error) {
            onActionComplete(error.message, 'error');
        }
    };

    return (
        <tr className="actionable-row">
            <td className="filename-cell clickable-filename" data-label="Имя файла" onClick={() => onOpenImageModal(item.IMG)}>
                {item.IMG}
            </td>
            <td data-label="Распознанный вид">{item.type}</td>
            <td className="action-cell" data-label="Паспорт особи">
                {!isActionPanelOpen ? (
                    <button className="action-button" onClick={() => setIsActionPanelOpen(true)}>Принять решение</button>
                ) : (
                    <div className="action-panel">
                        <p>Это новая особь или уже известная?</p>
                        <div className="action-buttons">
                            <button className="create-new-btn" onClick={() => onOpenAddModal(item)}>Создать новый паспорт</button>
                            <div className="assign-existing">
                                <select value={selectedPassportId} onChange={(e) => setSelectedPassportId(e.target.value)}>
                                    <option value="">-- Выбрать паспорт --</option>
                                    {relevantPassports.map(p => (<option key={p.id} value={p.id}>{p.name} (ID: {p.id})</option>))}
                                </select>
                                <button onClick={handleAssign} disabled={!selectedPassportId}>Присвоить</button>
                            </div>
                        </div>
                        <button className="cancel-btn" onClick={() => setIsActionPanelOpen(false)}>Отмена</button>
                    </div>
                )}
            </td>
        </tr>
    );
};


export default function Result({ params, setParams }) {
    const [showPassportModal, setShowPassportModal] = useState(false);
    const [currentPassportInfo, setCurrentPassportInfo] = useState(null);
    const [isInfoLoading, setIsInfoLoading] = useState(false);
    const [cordsHistory, setCordsHistory] = useState([]);
    const [galleryPhotos, setGalleryPhotos] = useState([]);
    const [currentPhotoIndex, setCurrentPhotoIndex] = useState(0);
    const [allPassports, setAllPassports] = useState([]);
    const [openAddModal, setOpenAddModal] = useState(false);
    const [itemForNewPassport, setItemForNewPassport] = useState(null);
    const [inputs, setInputs] = useState({ age: '', gender: '', name: '' });
    const [notification, setNotification] = useState({ show: false, message: '', type: '' });

    // Состояние для нового модального окна с изображением
    const [showImageModal, setShowImageModal] = useState(false);
    const [selectedImage, setSelectedImage] = useState('');

    const COLORS_pie = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#AF19FF', '#FF4560'];

    const showNotification = (message, type) => {
        setNotification({ show: true, message, type });
        setTimeout(() => {
            setNotification({ show: false, message: '', type: '' });
        }, 3000);
    };

    const fetchAllPassports = useCallback(async () => {
        try {
            const response = await fetch(`${API_URL}/all_passports`);
            if (!response.ok) throw new Error('Ошибка загрузки паспортов');
            const data = await response.json();
            setAllPassports(data);
        } catch (error) {
            console.error("Не удалось загрузить паспорта для присвоения:", error);
        }
    }, []);

    useEffect(() => {
        fetchAllPassports();
    }, [fetchAllPassports]);

    const handleActionComplete = (message, type, updateInfo) => {
        showNotification(message, type);
        fetchAllPassports();
        if (updateInfo && updateInfo.img && updateInfo.passportId) {
            const updatedPreds = params.params.pred.map(p => {
                if (p.IMG === updateInfo.img) {
                    return { ...p, passport: updateInfo.passportId };
                }
                return p;
            });
            setParams(currentParams => ({
                ...currentParams,
                params: {
                    ...currentParams.params,
                    pred: updatedPreds,
                    coordinates: currentParams.params.coordinates,
                }
            }));
        }
    };

    const handleOpenPassportModal = async (passportId) => {
        if (!passportId) return;
        setCurrentPassportInfo(null);
        setCordsHistory([]);
        setGalleryPhotos([]);
        setCurrentPhotoIndex(0);
        setIsInfoLoading(true);
        setShowPassportModal(true);
        try {
            const [passportResponse, historyResponse, photosResponse] = await Promise.all([
                fetch(`${API_URL}/get_passport/${passportId}`),
                fetch(`${API_URL}/passport/${passportId}/history`),
                fetch(`${API_URL}/passport/${passportId}/photos`)
            ]);
            if (!passportResponse.ok) throw new Error('Паспорт не найден');
            const passportData = await passportResponse.json();
            setCurrentPassportInfo(passportData);
            if (historyResponse.ok) {
                const historyData = await historyResponse.json();
                setCordsHistory(historyData);
            }
            if (photosResponse.ok) {
                const photoData = await photosResponse.json();
                const allPhotos = [
                    passportData.image_preview,
                    ...photoData.filter(p => p !== passportData.image_preview)
                ].filter(Boolean);
                setGalleryPhotos(allPhotos);
            }
        } catch (error) {
            console.error("Ошибка при получении данных паспорта:", error);
            showNotification(error.message, 'error');
            setShowPassportModal(false);
        } finally {
            setIsInfoLoading(false);
        }
    };

    const handleClosePassportModal = () => {
        setShowPassportModal(false);
    };

    // Функции для управления модальным окном с фото
    const handleOpenImageModal = (imageName) => {
        setSelectedImage(imageName);
        setShowImageModal(true);
    };

    const handleCloseImageModal = () => {
        setShowImageModal(false);
        setSelectedImage('');
    };

    const handlePhotoChange = (direction) => {
        if (galleryPhotos.length <= 1) return;
        if (direction === 'next') {
            setCurrentPhotoIndex((prevIndex) => (prevIndex + 1) % galleryPhotos.length);
        } else if (direction === 'prev') {
            setCurrentPhotoIndex((prevIndex) => (prevIndex - 1 + galleryPhotos.length) % galleryPhotos.length);
        }
    };

    const handleOpenAddModal = (item) => {
        setItemForNewPassport(item);
        setOpenAddModal(true);
    };

    const handleAddPassportSubmit = async (e) => {
        e.preventDefault();
        const originalCoords = params.params.coordinates || '0.0,0.0';
        const [lat, lng] = originalCoords.split(',');

        const formData = new FormData();
        formData.append('image_name', itemForNewPassport.IMG);
        Object.entries(inputs).forEach(([key, value]) => formData.append(key, value));
        formData.append('cords_sd', lat.trim());
        formData.append('cords_vd', lng.trim());

        try {
            const response = await fetch(`${API_URL}/create_passport_from_upload/`, { method: 'POST', body: formData });
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Не удалось создать паспорт.');
            }
            const newPassport = await response.json();
            setOpenAddModal(false);
            handleActionComplete('Паспорт успешно создан!', 'success', {
                img: itemForNewPassport.IMG,
                passportId: newPassport.id
            });
            setInputs({ age: '', gender: '', name: '' });
        } catch (error) {
            console.error(error);
            showNotification(`Ошибка: ${error.message}`, 'error');
        }
    };

    const handleInputChange = (e) => setInputs({ ...inputs, [e.target.name]: e.target.value });

    if (!params || !params.params) {
        return (
            <div className="result-page-container">
                <p>Нет данных для отображения.</p>
                <button className="finish-button" onClick={() => setParams({ "results": 0 })}>Новая обработка</button>
            </div>
        );
    }

    const { pred = [], diagram = {} } = params.params;
    const data_pie = Object.entries(diagram).map(([key, value]) => ({ name: key, value: value }));
    const totalPhotos = pred.length;
    const totalAnimals = data_pie.reduce((sum, entry) => sum + entry.value, 0);

    return (
        <>
            <div className={`notification-banner ${notification.show ? 'show' : ''} ${notification.type}`}>
                {notification.message}
            </div>

            <div className="result-page-container">
                <div className="result-header">
                    <div className="name_of_block">Результат обработки данных</div>
                    <div className="finish-button" onClick={() => setParams({ "results": 0 })}>Закончить просмотр</div>
                </div>

                <div className="stats-container">
                    <div className="stat-item">
                        <span className="stat-value">{totalPhotos}</span>
                        <span className="stat-label">Фото обработано</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-value">{totalAnimals}</span>
                        <span className="stat-label">Всего животных</span>
                    </div>
                    <div className="stat-item">
                        <span className="stat-value">{Object.keys(diagram).length}</span>
                        <span className="stat-label">Уникальных видов</span>
                    </div>
                </div>

                <div className="work_answer">
                    <div className="diagram">
                        <div className="pie-chart-name">Соотношение видов животных</div>
                        <PieChart width={350} height={300}>
                          <Pie
                            data={data_pie} cx="50%" cy="50%"
                            labelLine={false} outerRadius={80}
                            fill="#8884d8" dataKey="value"
                          >
                            {data_pie.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS_pie[index % COLORS_pie.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </PieChart>
                    </div>
                    <div className="table">
                        <div className='pie-chart-name'>Отчет о всех найденных животных</div>
                        <div className="table-wrapper">
                            <table className="results-table">
                                <thead>
                                    <tr>
                                        <th>Имя файла</th>
                                        <th>Вид</th>
                                        <th>Паспорт особи</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {pred.map((item) => {
                                        const isRareAndUnidentified = RARE_ANIMALS_LIST.includes(item.type) && item.passport === null;
                                        if (isRareAndUnidentified) {
                                            return <ActionableRow
                                                        key={item.IMG}
                                                        item={item}
                                                        allPassports={allPassports}
                                                        onActionComplete={handleActionComplete}
                                                        onOpenAddModal={handleOpenAddModal}
                                                        coordinates={params.params.coordinates}
                                                        onOpenImageModal={handleOpenImageModal}
                                                    />;
                                        }
                                        return (
                                            <tr key={item.IMG}>
                                                <td className="filename-cell clickable-filename" data-label="Имя файла" onClick={() => handleOpenImageModal(item.IMG)}>
                                                    {item.IMG}
                                                </td>
                                                <td data-label="Распознанный вид">{item.type}</td>
                                                <td data-label="Паспорт особи"
                                                    className={item.passport ? 'underline' : ''}
                                                    onClick={() => handleOpenPassportModal(item.passport)}
                                                >
                                                    {item.passport ? `Открыть (ID: ${item.passport})` : 'Отсутствует'}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <div className={`result-passport-modal ${showPassportModal ? 'visible' : ''}`}>
                <div className="animal-passport-wrapper">
                    <div className="close" onClick={handleClosePassportModal}>&times;</div>
                    {isInfoLoading ? (
                        <div className="loading-message">Загрузка данных...</div>
                    ) : currentPassportInfo ? (
                        <>
                            <div className="passport-modal-name">{currentPassportInfo.name || 'Паспорт животного'}</div>
                            <div className="photo-gallery">
                                <div className="gallery-image">
                                    {galleryPhotos.length > 0 && (
                                        <img src={`${API_URL}/image/passport/${galleryPhotos[currentPhotoIndex]}`} alt={`${currentPassportInfo.name} - фото ${currentPhotoIndex + 1}`} />
                                    )}
                                </div>
                                {galleryPhotos.length > 1 && (
                                    <div className="gallery-controls">
                                        <button onClick={() => handlePhotoChange('prev')} className="gallery-arrow">‹</button>
                                        <span className="gallery-counter">{currentPhotoIndex + 1} / {galleryPhotos.length}</span>
                                        <button onClick={() => handlePhotoChange('next')} className="gallery-arrow">›</button>
                                    </div>
                                )}
                            </div>
                            <div className="passport-modal-items">
                                <div className="passport-item"><label>Имя:</label> {currentPassportInfo.name}</div>
                                <div className="passport-item"><label>Вид:</label> {currentPassportInfo.type}</div>
                                <div className="passport-item"><label>ID:</label> {currentPassportInfo.id}</div>
                                <div className="passport-item"><label>Пол:</label> {currentPassportInfo.gender}</div>
                                <div className="passport-item"><label>Возраст:</label> {currentPassportInfo.age}</div>
                            </div>
                            <div className="passport-history">
                                <h4>История перемещений:</h4>
                                {cordsHistory.length > 0 ? (
                                    <ul>
                                        {cordsHistory.map((record, index) => (
                                            <li key={index}>{new Date(record.date).toLocaleDateString()}: {record.coordinates}</li>
                                        ))}
                                    </ul>
                                ) : (<p>История перемещений отсутствует.</p>)}
                            </div>
                        </>
                    ) : (
                         <div className="error-message">Не удалось загрузить данные.</div>
                    )}
                </div>
            </div>

            <div className={`modal-add-passport ${openAddModal ? 'visible' : ''}`}>
                 <div className="add-passport-container">
                    <div className="close" onClick={() => setOpenAddModal(false)}>&times;</div>
                    <h4 className="modal-title">Создание паспорта для: {itemForNewPassport?.type}</h4>
                    <form className="inputs_passport" onSubmit={handleAddPassportSubmit}>
                        <div className="add_name"><label>Имя</label><input type="text" name="name" value={inputs.name} onChange={handleInputChange} required /></div>
                        <div className="add_name"><label>Возраст</label><input type="number" name="age" value={inputs.age} onChange={handleInputChange} required /></div>
                        <div className="add_name"><label>Пол</label><input type="text" name="gender" value={inputs.gender} onChange={handleInputChange} required /></div>
                        <p className="modal-info">Фото и координаты будут взяты из исходного снимка.</p>
                        <button type="submit">Создать паспорт</button>
                    </form>
                </div>
            </div>

            <div className={`image-preview-modal ${showImageModal ? 'visible' : ''}`} onClick={handleCloseImageModal}>
                <div className="image-preview-wrapper" onClick={(e) => e.stopPropagation()}>
                    <div className="close" onClick={handleCloseImageModal}>&times;</div>
                    {selectedImage && (
                         <img src={`${API_URL}/image/passport/${selectedImage}`} alt={`Preview of ${selectedImage}`} />
                    )}
                </div>
            </div>
        </>
    );
}
