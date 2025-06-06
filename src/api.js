// src/api.js
// Handles all API calls for posts and authentication
import { groth16 } from "snarkjs";
import { get, set } from "idb-keyval";

async function loadZkey() {
  let cached = await get("circuit_final.zkey");

  if (cached) {
    console.log("Loaded .zkey from IndexedDB cache");
    return new Uint8Array(
      cached instanceof ArrayBuffer ? cached : cached.buffer
    );
  }

  console.log("Fetching .zkey from network...");
  const response = await fetch(
    `${import.meta.env.BASE_URL}keys/circuit_final.zkey`
  );
  const arrayBuffer = await response.arrayBuffer();

  await set("circuit_final.zkey", arrayBuffer);
  return new Uint8Array(arrayBuffer);
}

async function loadWasm() {
  let cached = await get("auth.wasm");

  if (cached) {
    console.log("Loaded .wasm from IndexedDB cache");
    return new Uint8Array(
      cached instanceof ArrayBuffer ? cached : cached.buffer
    );
  }

  const response = await fetch(
    `${import.meta.env.BASE_URL}circuits/auth_js/auth.wasm`
  );
  const buffer = await response.arrayBuffer();

  await set("auth.wasm", buffer);
  return new Uint8Array(buffer);
}

function getMemoryUsage() {
  if (performance.memory) {
    return {
      heapUsedMB: (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2),
      totalHeapMB: (performance.memory.totalJSHeapSize / 1024 / 1024).toFixed(
        2
      ),
      jsHeapLimitMB: (performance.memory.jsHeapSizeLimit / 1024 / 1024).toFixed(
        2
      ),
    };
  } else {
    return {
      heapUsedMB: "unsupported",
      totalHeapMB: "unsupported",
      jsHeapLimitMB: "unsupported",
    };
  }
}

/**
 * Converts a hex string to a bit array of specified length
 * @param {string} hexStr - Hex string (with or without 0x prefix)
 * @param {number} bitLength - Length of the output bit array
 * @returns {Array<number>} - Array of bits (0s and 1s)
 */
function hexToBits(hexStr, bitLength = 256) {
  // Remove 0x prefix if present
  const cleanHex = hexStr.startsWith("0x") ? hexStr.slice(2) : hexStr;

  // Make sure we have enough characters to extract the requested bits
  const requiredHexChars = Math.ceil(bitLength / 4);
  const paddedHex = cleanHex.padStart(requiredHexChars, "0");

  // Convert to binary string
  let binStr = "";
  for (let i = 0; i < paddedHex.length; i++) {
    // Convert each hex character to 4 bits
    const hexChar = paddedHex[i];
    const nibble = parseInt(hexChar, 16);
    if (isNaN(nibble)) {
      throw new Error(`Invalid hex character: ${hexChar}`);
    }
    // Convert to 4-bit binary string and pad with leading zeros
    binStr += nibble.toString(2).padStart(4, "0");
  }

  // Trim or pad to exact length
  if (binStr.length > bitLength) {
    // Take only the last `bitLength` bits
    binStr = binStr.slice(-bitLength);
  } else if (binStr.length < bitLength) {
    // Pad with leading zeros
    binStr = binStr.padStart(bitLength, "0");
  }

  // Convert to array of 0s and 1s
  return Array.from(binStr).map((bit) => parseInt(bit, 10));
}

/**
 * Generates a ZKP proof that the user is in the allowed list.
 */
export async function generateZKPProof(
  privKey,
  merkleRoot,
  pathElements,
  pathIndices
) {
  const wasmPath = `${import.meta.env.BASE_URL}circuits/auth_js/auth.wasm`;
  const zkeyPath = `${import.meta.env.BASE_URL}keys/circuit_final.zkey`;

  try {
    const totalStart = performance.now();

    console.log("Preparing ZKP inputs...");

    const prepStart = performance.now();
    // Validate input formats
    if (!privKey || typeof privKey !== "string") {
      throw new Error(`Invalid privKey format: ${privKey}`);
    }
    if (!merkleRoot || typeof merkleRoot !== "string") {
      throw new Error(`Invalid merkleRoot format: ${merkleRoot}`);
    }
    if (!Array.isArray(pathElements) || pathElements.length === 0) {
      throw new Error(`Invalid pathElements: ${JSON.stringify(pathElements)}`);
    }
    if (!Array.isArray(pathIndices) || pathIndices.length === 0) {
      throw new Error(`Invalid pathIndices: ${JSON.stringify(pathIndices)}`);
    }

    // Ensure pathElements and pathIndices have the same length
    if (pathElements.length !== pathIndices.length) {
      throw new Error(
        `Path elements (${pathElements.length}) and indices (${pathIndices.length}) must have same length`
      );
    }

    // Convert inputs to bit arrays as expected by the circuit
    const privKeyBits = hexToBits(privKey, 256);
    const rootBits = hexToBits(merkleRoot, 256);

    // Convert path elements to bit arrays
    const pathBits = pathElements.map((element) => hexToBits(element, 256));

    // Prepare input for the circuit
    const input = {
      leaf: privKeyBits, // bit array of private key
      root: rootBits, // bit array of merkle root
      path: pathBits, // array of bit arrays
      direction: pathIndices, // array of 0s and 1s indicating left/right
    };

    const prepEnd = performance.now();

    console.log("Generating ZKP proof with formatted inputs...");

    // Load the zkey and wasm files
    const loadStart = performance.now();
    const zkeyBuffer = await loadZkey();
    const wasmBuffer = await loadWasm();
    const loadEnd = performance.now();

    const proofStart = performance.now();
    const { proof, publicSignals } = await groth16.fullProve(
      input,
      wasmBuffer,
      zkeyBuffer
    );
    const mem = getMemoryUsage();
    console.log(
      `🧠 Memory usage: ${mem.heapUsedMB} MB used / ${mem.totalHeapMB} MB total`
    );

    const proofEnd = performance.now();

    // if (!proof) {
    //   throw new Error("ZKP proof generation failed: No proof returned");
    // }
    // if (!publicSignals || publicSignals.length === 0) {
    //   throw new Error("ZKP proof generation failed: No public signals returned");
    // }
    // console.log("ZKP proof and public signals generated successfully");

    const totalEnd = performance.now();
    console.log("Public Signals:", JSON.stringify(publicSignals));
    console.dir(publicSignals);
    return {
      auth: { proof, publicSignals },
      timings: {
        prepareTime: prepEnd - prepStart,
        loadTime: loadEnd - loadStart,
        proofTime: proofEnd - proofStart,
        totalTime: totalEnd - totalStart,
      }, memory: mem,
    };
  } catch (error) {
    console.error("Error generating ZKP proof:", error);
    throw new Error(`ZKP proof generation failed: ${error.message}`);
  }
}

/**
 * Fetch a batch of posts from the backend API.
 * @param {Object} params - Query parameters (topics, hashtags, limit, etc.)
 * @param {Object} [auth] - Optional authentication object (e.g., ZKP proof)
 * @returns {Promise<Object>} - The API response JSON
 */
export async function fetchPosts(params = {}, auth = null) {
  const url = new URL(
    "https://manatee-living-legally.ngrok-free.app/api/posts"
  );
  Object.entries(params).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      url.searchParams.append(key, value.join(","));
    } else {
      url.searchParams.append(key, value);
    }
  });

  const fetchOptions = {
    method: "GET",
    headers: {
      "ngrok-skip-browser-warning": "true",
    },
  };

  // If ZKP auth is provided, add it to headers or as needed
  if (auth && auth.proof) {
    fetchOptions.headers["X-ZKP-Proof"] = JSON.stringify(auth.proof);
    fetchOptions.headers["X-ZKP-PublicSignals"] = JSON.stringify(
      auth.publicSignals
    );
  }

  const res = await fetch(url.toString(), fetchOptions);
  const text = await res.text();

  try {
    return JSON.parse(text);
  } catch (err) {
    console.log("🔍 Raw server response:", text);
    throw new Error("Server returned non-JSON response");
  }
}

// You can add more API functions here, e.g., for authentication, user profile, etc.
