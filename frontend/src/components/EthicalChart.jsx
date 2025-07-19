/**
 * EthicalChart.jsx - Multidimensional Ethical Evaluation Visualization
 * 
 * Phase 4: Heat-Map Visualization Implementation
 * Specification: Sharp rectangles, solid fills, SVG-based for scalability
 * Color scheme: Red (<0.20), Orange (0.20-0.40), Yellow (0.40-0.60), Green (0.60-0.90), Blue (≥0.90)
 * Accessibility: ARIA labels, RTL support, WCAG AA compliance
 */

import React, { useState } from 'react';

/**
 * Color mapping function for ethical scores
 * @param {number} score - Ethical score (0-1)
 * @returns {string} - Hex color code
 */
function getEthicalColor(score) {
  if (score < 0.20) return '#ef4444'; // Red - Major violation
  if (score < 0.40) return '#f97316'; // Orange - Moderate violation
  if (score < 0.60) return '#eab308'; // Yellow - Minor concern
  if (score < 0.90) return '#22c55e'; // Green - Ethical
  return '#3b82f6'; // Blue - Highly ethical
}

/**
 * Get color description for accessibility
 * @param {number} score - Ethical score (0-1)
 * @returns {string} - Human-readable color description
 */
function getColorDescription(score) {
  if (score < 0.20) return 'Major Violation (Red)';
  if (score < 0.40) return 'Moderate Violation (Orange)';
  if (score < 0.60) return 'Minor Concern (Yellow)';
  if (score < 0.90) return 'Ethical (Green)';
  return 'Highly Ethical (Blue)';
}

/**
 * Calculate grade from average score
 * @param {number} avgScore - Average score (0-1)
 * @returns {string} - Letter grade with percentage
 */
function calculateGrade(avgScore) {
  const percentage = Math.round(avgScore * 100);
  let grade = 'F';
  
  if (avgScore >= 0.97) grade = 'A+';
  else if (avgScore >= 0.93) grade = 'A';
  else if (avgScore >= 0.90) grade = 'A-';
  else if (avgScore >= 0.87) grade = 'B+';
  else if (avgScore >= 0.83) grade = 'B';
  else if (avgScore >= 0.80) grade = 'B-';
  else if (avgScore >= 0.77) grade = 'C+';
  else if (avgScore >= 0.73) grade = 'C';
  else if (avgScore >= 0.70) grade = 'C-';
  else if (avgScore >= 0.67) grade = 'D+';
  else if (avgScore >= 0.63) grade = 'D';
  else if (avgScore >= 0.60) grade = 'D-';
  
  return `${grade} (${percentage}%)`;
}

/**
 * Tooltip component for span details
 */
const SpanTooltip = ({ span, dimension, score, isVisible, position }) => {
  if (!isVisible) return null;
  
  return (
    <div
      className="absolute z-50 bg-gray-900 text-white p-2 rounded-md shadow-lg border border-gray-600 text-sm"
      style={{
        left: position.x,
        top: position.y - 60,
        pointerEvents: 'none'
      }}
    >
      <div className="font-semibold">{dimension} Dimension</div>
      <div>Score: {score.toFixed(3)}</div>
      <div>Status: {getColorDescription(score)}</div>
      <div className="text-xs mt-1 max-w-48 truncate">
        Span: "{span.text}"
      </div>
    </div>
  );
};

/**
 * Individual dimension row with rect visualization
 */
const DimensionRow = ({ dimension, spans, totalTextLength }) => {
  const [tooltip, setTooltip] = useState({ visible: false, span: null, position: { x: 0, y: 0 } });
  
  const handleMouseEnter = (span, event) => {
    setTooltip({
      visible: true,
      span: span,
      position: { x: event.clientX, y: event.clientY }
    });
  };
  
  const handleMouseLeave = () => {
    setTooltip({ visible: false, span: null, position: { x: 0, y: 0 } });
  };
  
  return (
    <div className="flex items-center mb-1 relative">
      <span className="w-8 text-sm font-medium text-gray-300 mr-2">{dimension}</span>
      <div className="flex-1 relative">
        <svg width="100%" height="24" className="border border-gray-600">
          {spans.map((span, idx) => {
            const score = span.scores[dimension] || 0;
            const startPercent = (span.span[0] / totalTextLength) * 100;
            const widthPercent = ((span.span[1] - span.span[0]) / totalTextLength) * 100;
            
            return (
              <rect
                key={idx}
                x={`${startPercent}%`}
                y="2"
                width={`${widthPercent}%`}
                height="20"
                fill={getEthicalColor(score)}
                stroke={span.uncertainty && span.uncertainty > 0.25 ? '#000000' : 'none'}
                strokeWidth={span.uncertainty && span.uncertainty > 0.25 ? '1' : '0'}
                onMouseEnter={(e) => handleMouseEnter({...span, dimension, score}, e)}
                onMouseLeave={handleMouseLeave}
                className="cursor-pointer"
                aria-label={`${dimension} Dimension: Score ${score.toFixed(3)} - ${getColorDescription(score)}`}
                role="img"
              />
            );
          })}
        </svg>
      </div>
      
      <SpanTooltip 
        span={tooltip.span} 
        dimension={dimension}
        score={tooltip.span?.score || 0}
        isVisible={tooltip.visible}
        position={tooltip.position}
      />
    </div>
  );
};

/**
 * Span type section (short/medium/long/stochastic)
 */
const SpanTypeSection = ({ type, data, totalTextLength }) => {
  if (!data || !data.spans || data.spans.length === 0) {
    return null;
  }
  
  const dimensions = ['V', 'A', 'C']; // Virtue, Autonomy, Consequentialist
  const avgScore = data.averageScore || 0;
  const grade = calculateGrade(avgScore);
  
  return (
    <div className="mb-6 bg-gray-800 p-4 rounded-lg border border-gray-600">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-lg font-semibold text-white capitalize">
          {type} Spans Analysis
        </h3>
        <div className="text-sm font-medium text-gray-300">
          Grade: <span className="text-white">{grade}</span>
        </div>
      </div>
      
      <div className="space-y-1">
        {dimensions.map(dimension => (
          <DimensionRow
            key={dimension}
            dimension={dimension}
            spans={data.spans}
            totalTextLength={totalTextLength}
          />
        ))}
      </div>
      
      {data.metadata && (
        <div className="mt-2 text-xs text-gray-400">
          Spans: {data.spans.length} | 
          Avg Score: {avgScore.toFixed(3)} |
          {data.metadata.dataset_source && ` Source: ${data.metadata.dataset_source}`}
        </div>
      )}
    </div>
  );
};

/**
 * Main EthicalChart component
 */
const EthicalChart = ({ data, className = '' }) => {
  if (!data || !data.evaluations) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg border border-gray-600">
        <div className="text-center text-gray-400">
          <div className="text-lg mb-2">📊 Heat-Map Visualization</div>
          <div className="text-sm">No evaluation data available</div>
        </div>
      </div>
    );
  }
  
  const spanTypes = ['short', 'medium', 'long', 'stochastic'];
  const totalTextLength = data.textLength || 100; // Fallback to prevent division by zero
  
  return (
    <div 
      className={`bg-gray-900 p-6 rounded-lg ${className}`} 
      role="region" 
      aria-label="Ethical Evaluation Heat Map"
      dir="auto"
    >
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white mb-2">
          📊 Multidimensional Ethical Evaluation Heat-Map
        </h2>
        <div className="text-sm text-gray-400">
          Interactive visualization across span granularities (V=Virtue, A=Autonomy, C=Consequentialist)
        </div>
      </div>
      
      {/* Color Legend */}
      <div className="mb-6 bg-gray-800 p-3 rounded-lg border border-gray-600">
        <div className="text-sm font-medium text-white mb-2">Color Scale:</div>
        <div className="flex flex-wrap gap-4 text-xs">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-red-500 mr-2 border border-gray-500"></div>
            <span className="text-gray-300">Major Violation (&lt;0.20)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-orange-500 mr-2 border border-gray-500"></div>
            <span className="text-gray-300">Moderate (0.20-0.40)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-yellow-500 mr-2 border border-gray-500"></div>
            <span className="text-gray-300">Minor Concern (0.40-0.60)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-green-500 mr-2 border border-gray-500"></div>
            <span className="text-gray-300">Ethical (0.60-0.90)</span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-blue-500 mr-2 border border-gray-500"></div>
            <span className="text-gray-300">Highly Ethical (≥0.90)</span>
          </div>
        </div>
      </div>
      
      {/* Span Type Sections */}
      <div className="space-y-4">
        {spanTypes.map(type => (
          <SpanTypeSection
            key={type}
            type={type}
            data={data.evaluations[type]}
            totalTextLength={totalTextLength}
          />
        ))}
      </div>
      
      {/* Overall Summary */}
      {data.overallGrades && (
        <div className="mt-6 bg-gray-800 p-4 rounded-lg border border-gray-600">
          <h3 className="text-lg font-semibold text-white mb-3">Overall Assessment</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            {spanTypes.map(type => (
              data.overallGrades[type] && (
                <div key={type} className="text-center">
                  <div className="text-gray-400 capitalize">{type}</div>
                  <div className="text-white font-medium">{data.overallGrades[type]}</div>
                </div>
              )
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default EthicalChart;