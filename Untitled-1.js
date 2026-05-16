/**
 * Розраховує щільність нормального розподілу для заданого x
 */
function calculateNormalPDF(x, mean, stdDev) {
  const variance = stdDev * stdDev;
  const exponent = Math.exp(-Math.pow(x - mean, 2) / (2 * variance));
  return (1 / Math.sqrt(2 * Math.PI * variance)) * exponent;
}

const mu = 1000;
const v_year = 0.123;
const sigma = 50;
const y_obs = 850; // Фактичне значення з журналу (Ledger)

const mean = mu * (1 - v_year);

// Отримуємо вірогідність (Likelihood)
const likelihood = calculateNormalPDF(y_obs, mean, sigma);

console.log(`Вірогідність спостереження: ${likelihood}`);