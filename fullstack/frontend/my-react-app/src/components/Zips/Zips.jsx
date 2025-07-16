import './zips.css';
import { useEffect, useState } from 'react';
import Result from '../ProcessPhoto/Result/Result';

const API_URL = import.meta.env.VITE_API_BASE_URL;

export default function Zips() {
    const [zipsData, setZipsData] = useState([]);
    const [selectedZipResults, setSelectedZipResults] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const formatDate = (dateString) => {
        const options = { year: 'numeric', month: 'long', day: 'numeric' };
        return new Date(dateString).toLocaleDateString('ru-RU', options);
    };

    useEffect(() => {
        const fetchZips = async () => {
            setLoading(true);
            try {
                const response = await fetch(`${API_URL}/all_zips`);
                if (!response.ok) {
                    throw new Error('Не удалось загрузить архив измерений.');
                }
                const data = await response.json();
                setZipsData(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        fetchZips();
    }, []);

    const handleZipClick = async (zipId) => {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/get_uploads/${zipId}`);
            if (!response.ok) {
                throw new Error('Не удалось загрузить детали этого измерения.');
            }
            const resultsData = await response.json();
            setSelectedZipResults({ 'results': 1, 'params': resultsData });
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleGoBack = () => {
        setSelectedZipResults(null);
        setError(null); // Сбрасываем ошибку при возврате
    };

    if (loading) {
        return <div className="loading-message">Загрузка архива...</div>;
    }

    if (error) {
        return <div className="error-message">Ошибка: {error} <button onClick={handleGoBack}>Назад</button></div>;
    }

    if (selectedZipResults) {
        return <Result params={selectedZipResults} setParams={handleGoBack} />;
    }

    return (
        <div className="zips-page-container">
            <div className="name-of-str">Архив измерений</div>
            <p className="zips-subtitle">Нажмите на любую запись, чтобы просмотреть детальный отчет.</p>
            <div className="table-container">
                <table className="archive-table">
                    <thead>
                        <tr>
                            <th>ID Загрузки</th>
                            <th>Дата</th>
                            <th>Координаты</th>
                            <th>Редких видов найдено</th>
                        </tr>
                    </thead>
                    <tbody>
                        {zipsData.map((item) => (
                            <tr key={item.id} onClick={() => handleZipClick(item.id)}>
                                <td>{item.id}</td>
                                <td>{formatDate(item.upload_date)}</td>
                                <td>{item.coordinates || 'Не указаны'}</td>
                                <td>{item.rare_animals_count}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
