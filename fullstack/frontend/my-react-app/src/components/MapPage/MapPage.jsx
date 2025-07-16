import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, Popup } from 'react-leaflet';
import './MapPage.css';

const API_URL = import.meta.env.VITE_API_BASE_URL;

export default function MapPage() {
    const [passports, setPassports] = useState([]);
    const [selectedPassportId, setSelectedPassportId] = useState('');
    const [path, setPath] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchPassports = async () => {
            try {
                const response = await fetch(`${API_URL}/all_passports`);
                if (!response.ok) throw new Error('Не удалось загрузить паспорта');
                const data = await response.json();
                setPassports(data);
            } catch (err) {
                setError(err.message);
            }
        };
        fetchPassports();
    }, []);

    useEffect(() => {
        if (!selectedPassportId) {
            setPath([]);
            return;
        }

        const fetchHistory = async () => {
            setLoading(true);
            setError('');
            try {
                const response = await fetch(`${API_URL}/passport/${selectedPassportId}/history`);
                if (!response.ok) throw new Error('Не удалось загрузить историю перемещений');
                const data = await response.json();

                // Преобразуем координаты из строки в массив чисел
                const formattedPath = data
                    .map(record => {
                        try {
                            const [lat, lng] = record.coordinates.split(',').map(Number);
                            return { pos: [lat, lng], date: new Date(record.date).toLocaleDateString() };
                        } catch (e) {
                            return null;
                        }
                    })
                    .filter(p => p && !isNaN(p.pos[0]) && !isNaN(p.pos[1]));

                setPath(formattedPath);

            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchHistory();
    }, [selectedPassportId]);

    const position = path.length > 0 ? path[0].pos : [54.5, 105.0];

    return (
        <div className="map-page-container">
            <div className="map-controls">
                <h2>Карта перемещений</h2>
                <p>Выберите паспорт особи, чтобы увидеть историю ее перемещений.</p>
                <select
                    value={selectedPassportId}
                    onChange={(e) => setSelectedPassportId(e.target.value)}
                    className="passport-select"
                >
                    <option value="">-- Выберите животное --</option>
                    {passports.map(p => (
                        <option key={p.id} value={p.id}>{p.name} (ID: {p.id})</option>
                    ))}
                </select>
                {loading && <p className="loading-text">Загрузка данных...</p>}
                {error && <p className="error-text">{error}</p>}
            </div>

            <MapContainer center={position} zoom={5} className="map-view" key={selectedPassportId}>
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {path.length > 0 && (
                    <>
                        <Polyline positions={path.map(p => p.pos)} color="blue" />
                        {path.map((point, index) => (
                            <Marker key={index} position={point.pos}>
                                <Popup>
                                    Точка {index + 1} <br /> {point.date}
                                </Popup>
                            </Marker>
                        ))}
                    </>
                )}
            </MapContainer>
        </div>
    );
}
