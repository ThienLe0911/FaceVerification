import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { UserPlus, Search, Settings, Eye } from 'lucide-react';

const Header = () => {
  const location = useLocation();

  const navItems = [
    { 
      path: '/enroll', 
      label: 'Enroll PersonA', 
      icon: UserPlus,
      description: 'Upload 20-40 ảnh để tạo gallery'
    },
    { 
      path: '/verify', 
      label: 'Verify PersonA', 
      icon: Search,
      description: 'Kiểm tra ảnh có PersonA không'
    },
    { 
      path: '/settings', 
      label: 'Cài đặt', 
      icon: Settings,
      description: 'Điều chỉnh threshold'
    },
  ];

  return (
    <header className="bg-white shadow-lg border-b border-gray-200">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-primary-600 rounded-lg">
              <Eye className="h-6 w-6 text-white" />
            </div>
            <div>
              <div className="text-xl font-bold text-gray-900">
                Face Verification
              </div>
              <div className="text-xs text-gray-500 hidden sm:block">
                PersonA Recognition System
              </div>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex space-x-1">
            {navItems.map(({ path, label, icon: Icon, description }) => (
              <Link
                key={path}
                to={path}
                className={`group flex flex-col items-center px-4 py-3 rounded-lg transition-all duration-200 ${
                  location.pathname === path
                    ? 'bg-primary-100 text-primary-700 shadow-sm'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <Icon className="h-5 w-5" />
                  <span className="font-medium">{label}</span>
                </div>
                <div className="text-xs text-gray-500 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  {description}
                </div>
              </Link>
            ))}
          </nav>

          {/* Mobile Navigation */}
          <nav className="md:hidden flex space-x-2">
            {navItems.map(({ path, icon: Icon }) => (
              <Link
                key={path}
                to={path}
                className={`flex items-center justify-center w-10 h-10 rounded-lg transition-colors duration-200 ${
                  location.pathname === path
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Icon className="h-5 w-5" />
              </Link>
            ))}
          </nav>
        </div>

        {/* Sub-navigation hint */}
        <div className="hidden md:block pb-2">
          <div className="text-center">
            <p className="text-sm text-gray-500">
              <span className="font-medium text-primary-600">Enroll:</span> Tạo gallery PersonA → 
              <span className="font-medium text-green-600 mx-2">Verify:</span> Kiểm tra ảnh có PersonA không
            </p>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;