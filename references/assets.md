# Asset Specifications

## Table of Contents
1. [Forex Majors](#forex-majors)
2. [Gold (XAUUSD)](#gold-xauusd)
3. [Session Times](#session-times)
4. [Spread Analysis](#spread-analysis)
5. [Broker Considerations](#broker-considerations)

---

## Forex Majors

### EURUSD (Euro/US Dollar)
```
Point Value:      $10 per pip per standard lot
Pip Size:         0.0001
Typical Spread:   0.5 - 1.5 pips (ECN)
Avg Daily Range:  60-100 pips
Best Sessions:    London, London/NY overlap
Volatility:       Medium
Correlation:      Inverse to DXY, positive to GBPUSD
```

**Scalping Considerations:**
- Tightest spreads during London session (08:00-12:00 UTC)
- Avoid 5 minutes before/after major news (ECB, Fed, NFP)
- Minimum viable target: 5 pips (after spread)

### GBPUSD (British Pound/US Dollar)
```
Point Value:      $10 per pip per standard lot
Pip Size:         0.0001
Typical Spread:   1.0 - 2.5 pips (ECN)
Avg Daily Range:  80-150 pips
Best Sessions:    London, London/NY overlap
Volatility:       High
Correlation:      Positive to EURUSD
```

**Scalping Considerations:**
- Higher volatility requires wider stops than EURUSD
- Significant moves during UK data releases (09:00 UTC)
- Flash crash risk during illiquid hours (20:00-00:00 UTC)
- Minimum viable target: 8-10 pips

### USDJPY (US Dollar/Japanese Yen)
```
Point Value:      ~$9.30 per pip per standard lot (varies with rate)
Pip Size:         0.01
Typical Spread:   0.8 - 1.8 pips (ECN)
Avg Daily Range:  50-90 pips
Best Sessions:    Asian, NY
Volatility:       Medium
Correlation:      Positive to US yields, inverse to Gold
```

**Scalping Considerations:**
- Active during Asian session (00:00-08:00 UTC)
- Responds strongly to US Treasury yields
- BOJ interventions can cause 100+ pip moves instantly
- Minimum viable target: 6-8 pips

### USDCHF (US Dollar/Swiss Franc)
```
Point Value:      ~$10.50 per pip per standard lot (varies)
Pip Size:         0.0001
Typical Spread:   1.0 - 2.0 pips (ECN)
Avg Daily Range:  50-80 pips
Best Sessions:    London
Volatility:       Low-Medium
Correlation:      Inverse to EURUSD, safe haven
```

**Scalping Considerations:**
- Lower volatility, requires patience
- Safe haven flows during risk-off events
- SNB has history of sudden interventions
- Minimum viable target: 5-7 pips

### AUDUSD (Australian Dollar/US Dollar)
```
Point Value:      $10 per pip per standard lot
Pip Size:         0.0001
Typical Spread:   0.8 - 1.8 pips (ECN)
Avg Daily Range:  60-100 pips
Best Sessions:    Asian, London/NY overlap
Volatility:       Medium-High
Correlation:      Positive to commodities, risk-on
```

**Scalping Considerations:**
- Active during Asian session (22:00-08:00 UTC)
- Sensitive to Chinese economic data
- Risk-on/risk-off dynamics important
- Minimum viable target: 6-8 pips

### NZDUSD (New Zealand Dollar/US Dollar)
```
Point Value:      $10 per pip per standard lot
Pip Size:         0.0001
Typical Spread:   1.0 - 2.5 pips (ECN)
Avg Daily Range:  50-80 pips
Best Sessions:    Asian, London
Volatility:       Medium
Correlation:      Highly positive to AUDUSD
```

### USDCAD (US Dollar/Canadian Dollar)
```
Point Value:      ~$7.60 per pip per standard lot (varies)
Pip Size:         0.0001
Typical Spread:   1.2 - 2.5 pips (ECN)
Avg Daily Range:  60-100 pips
Best Sessions:    NY
Volatility:       Medium
Correlation:      Inverse to oil prices
```

---

## Gold (XAUUSD)

### Specifications
```
Contract Size:    100 oz per standard lot
Point Size:       0.01 (1 cent)
Tick Value:       $1 per tick per lot
Typical Spread:   15-35 points (ECN)
Avg Daily Range:  $15-40 ($1500-4000 in points)
Best Sessions:    London, NY
Volatility:       Very High
```

### Risk Parameters
| Risk Level | Max Position | Stop Loss | Target |
|------------|--------------|-----------|--------|
| Conservative | 0.1 lot | $10 (1000 pts) | $15-20 |
| Moderate | 0.3 lot | $8 (800 pts) | $12-18 |
| Aggressive | 0.5 lot | $5 (500 pts) | $8-15 |

### Trading Considerations

**Volatility Adjustment:**
```mql5
// Gold requires larger stops than Forex
double GetGoldStopLoss(double atr, double multiplier = 2.0) {
    // ATR is typically $10-20 for Gold on H1
    // Minimum SL should be at least $3-5
    double minSL = 300;  // 300 points = $3

    double calculatedSL = atr * multiplier / _Point;
    return MathMax(minSL, calculatedSL);
}
```

**Session Behavior:**
- Asian (00:00-08:00 UTC): Lower volatility, range-bound
- London (08:00-16:00 UTC): Breakout potential, trending
- NY (13:00-21:00 UTC): Highest volatility, news-driven
- London/NY overlap (13:00-16:00 UTC): Maximum liquidity

**Correlation Awareness:**
- Inverse to DXY (Dollar Index)
- Inverse to real yields (TIPS)
- Positive to inflation expectations
- Safe haven during market stress

**News Events Impact:**
| Event | Expected Move | Avoid Window |
|-------|---------------|--------------|
| NFP | $10-30 | ±30 minutes |
| CPI | $10-25 | ±30 minutes |
| FOMC | $15-40 | ±60 minutes |
| Fed Speak | $5-15 | ±15 minutes |

### Gold-Specific MQL5 Code

```mql5
bool IsGoldSymbol() {
    string symbol = _Symbol;
    return (StringFind(symbol, "XAU") >= 0 ||
            StringFind(symbol, "GOLD") >= 0);
}

double GetGoldLotSize(double riskPercent, double slPoints) {
    // Gold tick value is $1 per point per lot
    // Different calculation than Forex

    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = balance * riskPercent / 100.0;

    // For Gold: 1 point = $0.01, so 100 points = $1
    // 1 lot = 100 oz, tick value typically $1
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

    double pointValue = tickValue / tickSize * _Point;
    double lots = riskAmount / (slPoints * pointValue);

    // Gold often has 0.01 lot minimum
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lots = MathFloor(lots / lotStep) * lotStep;
    return MathMax(minLot, lots);
}
```

---

## Session Times

### Session Schedule (UTC)

```
Sydney:     22:00 - 07:00
Tokyo:      00:00 - 09:00
London:     08:00 - 17:00
New York:   13:00 - 22:00

Overlaps:
Tokyo/London:    08:00 - 09:00 (low overlap)
London/NY:       13:00 - 17:00 (highest volume)
```

### MQL5 Session Filter

```mql5
enum ENUM_SESSION {
    SESSION_SYDNEY,
    SESSION_TOKYO,
    SESSION_LONDON,
    SESSION_NEWYORK,
    SESSION_OVERLAP_LONDON_NY
};

bool IsInSession(ENUM_SESSION session) {
    MqlDateTime now;
    TimeToStruct(TimeGMT(), now);
    int hour = now.hour;

    switch(session) {
        case SESSION_SYDNEY:
            return (hour >= 22 || hour < 7);

        case SESSION_TOKYO:
            return (hour >= 0 && hour < 9);

        case SESSION_LONDON:
            return (hour >= 8 && hour < 17);

        case SESSION_NEWYORK:
            return (hour >= 13 && hour < 22);

        case SESSION_OVERLAP_LONDON_NY:
            return (hour >= 13 && hour < 17);
    }

    return false;
}

// Session volatility multiplier
double GetSessionVolatilityMultiplier() {
    if(IsInSession(SESSION_OVERLAP_LONDON_NY)) return 1.3;
    if(IsInSession(SESSION_LONDON)) return 1.1;
    if(IsInSession(SESSION_NEWYORK)) return 1.0;
    if(IsInSession(SESSION_TOKYO)) return 0.8;
    if(IsInSession(SESSION_SYDNEY)) return 0.6;
    return 0.7;  // Inter-session
}
```

### Best Times by Pair

| Pair | Best Session | Peak Hours (UTC) | Avoid |
|------|--------------|------------------|-------|
| EURUSD | London/NY | 13:00-16:00 | 20:00-06:00 |
| GBPUSD | London | 08:00-16:00 | 21:00-07:00 |
| USDJPY | Asian/NY | 00:00-03:00, 13:00-16:00 | 06:00-08:00 |
| AUDUSD | Asian | 00:00-06:00 | 18:00-22:00 |
| XAUUSD | London/NY | 08:00-16:00 | 00:00-06:00 |

---

## Spread Analysis

### Spread Monitoring

```mql5
class CSpreadAnalyzer {
private:
    double spreads[];
    int maxSamples;
    int currentIndex;

public:
    CSpreadAnalyzer(int samples = 1000) {
        maxSamples = samples;
        ArrayResize(spreads, samples);
        ArrayInitialize(spreads, 0);
        currentIndex = 0;
    }

    void AddSample() {
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        spreads[currentIndex] = (ask - bid) / _Point;
        currentIndex = (currentIndex + 1) % maxSamples;
    }

    double GetAverageSpread() {
        double sum = 0;
        int count = 0;
        for(int i = 0; i < maxSamples; i++) {
            if(spreads[i] > 0) {
                sum += spreads[i];
                count++;
            }
        }
        return count > 0 ? sum / count : 0;
    }

    double GetMaxSpread() {
        double maxSpread = 0;
        for(int i = 0; i < maxSamples; i++) {
            if(spreads[i] > maxSpread) maxSpread = spreads[i];
        }
        return maxSpread;
    }

    bool IsSpreadAcceptable(double maxAllowed) {
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double current = (ask - bid) / _Point;

        return current <= maxAllowed;
    }
};
```

### Expected Spreads by Session

| Pair | Sydney | Tokyo | London | NY | Overlap |
|------|--------|-------|--------|----|---------|
| EURUSD | 1.5-3.0 | 1.0-2.0 | 0.5-1.2 | 0.6-1.5 | 0.5-1.0 |
| GBPUSD | 2.0-4.0 | 1.5-3.0 | 0.8-1.8 | 1.0-2.0 | 0.8-1.5 |
| USDJPY | 1.0-2.5 | 0.8-1.5 | 1.0-2.0 | 0.8-1.8 | 0.8-1.5 |
| XAUUSD | 30-60 | 20-40 | 15-30 | 15-30 | 15-25 |

---

## Broker Considerations

### ECN vs Market Maker

**ECN (Recommended for Scalping):**
- Raw spreads + commission
- Faster execution
- No dealing desk
- Better for tick-based strategies

**Market Maker:**
- Wider spreads, no commission
- May have execution delays
- Potential for requotes
- Not ideal for aggressive scalping

### Key Broker Metrics

```
Execution Speed:    < 50ms preferred
Slippage:           Track average deviation
Commission:         $3-7 per lot round trip
Minimum Lot:        0.01 preferred
Stop Level:         0 or minimal preferred
```

### Broker Validation in EA

```mql5
bool ValidateBrokerConditions() {
    // Check stop level
    int stopLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
    if(stopLevel > 10) {
        Print("Warning: High stop level: ", stopLevel, " points");
    }

    // Check freeze level
    int freezeLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
    if(freezeLevel > 5) {
        Print("Warning: High freeze level: ", freezeLevel, " points");
    }

    // Check execution mode
    ENUM_SYMBOL_TRADE_EXECUTION execMode =
        (ENUM_SYMBOL_TRADE_EXECUTION)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_EXEMODE);

    if(execMode == SYMBOL_TRADE_EXECUTION_MARKET) {
        Print("Market execution - slippage possible");
    } else if(execMode == SYMBOL_TRADE_EXECUTION_INSTANT) {
        Print("Instant execution - requotes possible");
    } else if(execMode == SYMBOL_TRADE_EXECUTION_EXCHANGE) {
        Print("Exchange execution - best for scalping");
    }

    return true;
}
```

### Recommended Broker Features

| Feature | Scalping | Intraday |
|---------|----------|----------|
| Execution | < 30ms | < 100ms |
| Spread | Variable ECN | Variable OK |
| Commission | < $4/lot | < $7/lot |
| Stop Level | 0-5 points | < 20 points |
| Leverage | 1:100+ | 1:50+ |
| Swap Free | Optional | Preferred |
