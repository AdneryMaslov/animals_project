import './navigation.css'
import { useState } from 'react'
import { Link } from 'react-router-dom';

export default function Navigation() {
    const [showMenu, setShowMenu] = useState(false);

    const toggleMenu = () => {
        setShowMenu(!showMenu);
    };
    const closeMenu = () => {
        setShowMenu(false);
    }

    return (
    <>
            <div className="navigation-container">
                <Link to="/" className="brand-name" onClick={closeMenu}>
                    SALMA'S
                </Link>
                <div className="nav-items">
                    <Link to="/map" className="nav-item">Карта перемещений</Link>
                    <Link to="/zips" className="nav-item">Архив измерений</Link>
                    <Link to="/animals" className="nav-item">Животные</Link>
                </div>
                <div className="burger" onClick={toggleMenu}>
                    <div className="burger-item"></div>
                    <div className="burger-item"></div>
                    <div className="burger-item"></div>
                </div>
            </div>

            <div className={`mobile-nav-overlay ${showMenu ? 'visible' : ''}`} onClick={closeMenu}>
                <div className="mobile-nav-items" onClick={(e) => e.stopPropagation()}>
                    <Link to="/zips" className="mobile-nav-item" onClick={closeMenu}>Архив измерений</Link>
                    <Link to="/animals" className="mobile-nav-item" onClick={closeMenu}>Животные</Link>
                    <Link to="/map" className="mobile-nav-item" onClick={closeMenu}>Карта перемещений</Link>
                </div>
            </div>
    </>
    )
}
