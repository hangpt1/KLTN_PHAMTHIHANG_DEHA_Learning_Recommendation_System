/**
 * E-Learning Recommendation System - Chart Utilities
 * Chart.js helper functions and initializations
 */

// Chart.js default configuration
Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#858796';

// Color palette
const chartColors = {
    primary: '#4e73df',
    success: '#1cc88a',
    info: '#36b9cc',
    warning: '#f6c23e',
    danger: '#e74a3b',
    secondary: '#858796',
    dark: '#2c3e50'
};

// Generate color array
function getColorArray(count) {
    const colors = [
        chartColors.primary,
        chartColors.success,
        chartColors.info,
        chartColors.warning,
        chartColors.danger,
        chartColors.secondary,
        '#8e44ad',
        '#27ae60',
        '#e67e22',
        '#3498db'
    ];
    return colors.slice(0, count);
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Create gradient for line charts
function createGradient(ctx, color) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, color + '40');
    gradient.addColorStop(1, color + '00');
    return gradient;
}

// Tooltip configuration
const tooltipConfig = {
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    titleFont: { size: 14, weight: 'bold' },
    bodyFont: { size: 12 },
    padding: 12,
    cornerRadius: 8,
    displayColors: true
};

// Common chart options
const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        tooltip: tooltipConfig
    }
};

console.log('E-Learning Charts JS loaded');
