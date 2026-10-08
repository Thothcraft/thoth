/** Thoth Cell mark — isometric cube, three sensing faces on a 3×3
 * lattice, nucleus where they meet. Dark-ops palette for the node
 * console; identical geometry to the site/portal/app mark. */
const LATTICE: Array<[number, number, number, number, boolean]> = [
  [24.5, 10.33, 47, 23.33, false], [17, 14.67, 39.5, 27.67, false],
  [39.5, 10.33, 17, 23.33, false], [47, 14.67, 24.5, 27.67, false],
  [9.5, 27.67, 32, 40.67, true], [9.5, 36.33, 32, 49.33, true],
  [17, 23.33, 17, 49.33, true], [24.5, 27.67, 24.5, 53.67, true],
  [32, 40.67, 54.5, 27.67, false], [32, 49.33, 54.5, 36.33, false],
  [39.5, 27.67, 39.5, 53.67, false], [47, 23.33, 47, 49.33, false],
];

export default function CellLogo({ size = 20, pulse = false }: { size?: number; pulse?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" role="img" aria-label="Thoth"
      style={{ display: 'inline-block', verticalAlign: 'middle', overflow: 'visible' }}>
      <polygon points="32,6 54.5,19 32,32 9.5,19" fill="#3a372e" />
      <polygon points="9.5,19 32,32 32,58 9.5,45" fill="#f4f1e9" />
      <polygon points="32,32 54.5,19 54.5,45 32,58" fill="#c96f3f" />
      {LATTICE.map(([x1, y1, x2, y2, onLeft], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
          stroke={onLeft ? 'rgba(17,17,15,0.3)' : 'rgba(244,241,233,0.3)'}
          strokeWidth="0.9" strokeLinecap="round" />
      ))}
      <polygon points="32,6 54.5,19 54.5,45 32,58 9.5,45 9.5,19"
        fill="none" stroke="#f4f1e9" strokeWidth="1.2" strokeLinejoin="round" />
      <circle cx="32" cy="32" r={pulse ? 7.5 : 0} fill="none" stroke="#f4f1e9"
        strokeWidth="1" opacity="0.5">
        {pulse && <animate attributeName="r" values="5;10" dur="1.6s" repeatCount="indefinite" />}
        {pulse && <animate attributeName="opacity" values="0.7;0" dur="1.6s" repeatCount="indefinite" />}
      </circle>
      <circle cx="32" cy="32" r="4" fill="#f4f1e9" />
    </svg>
  );
}
