import Navigation from './components/navigation/Navigation'
import ProcessPhoto from './components/ProcessPhoto/ProcessPhoto';
import Zips from './components/Zips/Zips';
import Animals from './components/animals/animals';
import MapPage from './components/MapPage/MapPage';
import "./style.css";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

export default function App() {
  return (
    <>
      <Router>
          <Navigation/>
          <main className="main-content-area">
            <Routes>
                <Route path="/" element={<ProcessPhoto />} />
                <Route path="/zips" element={<Zips />} />
                <Route path="/animals" element={<Animals />} />
                <Route path="/map" element={<MapPage />} />
            </Routes>
          </main>
      </Router>
    </>
  )
}
