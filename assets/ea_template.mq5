//+------------------------------------------------------------------+
//|                                              EA_Template.mq5     |
//|                                          Quant Trading MT5       |
//|                          Tick-Based Scalping/Intraday Template   |
//+------------------------------------------------------------------+
#property copyright "Quant Trading"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Risk Management ==="
input double   InpRiskPercent = 1.0;           // Risk per trade (%)
input double   InpMaxDailyDD = 3.0;            // Max daily drawdown (%)
input double   InpMaxTotalDD = 10.0;           // Max total drawdown (%)
input int      InpMaxPositions = 3;            // Max concurrent positions
input int      InpMaxSpread = 20;              // Max spread (points)

input group "=== Strategy Parameters ==="
input int      InpFastPeriod = 5;              // Fast period
input int      InpSlowPeriod = 20;             // Slow period
input double   InpZScoreEntry = 2.0;           // Z-Score entry threshold
input int      InpLookback = 100;              // Lookback period

input group "=== Take Profit & Stop Loss ==="
input double   InpSLMultiplier = 2.0;          // SL ATR multiplier
input double   InpTPMultiplier = 3.0;          // TP ATR multiplier
input bool     InpUseTrailing = true;          // Use trailing stop
input double   InpTrailMultiplier = 1.5;       // Trail ATR multiplier

input group "=== Session Filter ==="
input bool     InpUseLondon = true;            // Trade London session
input bool     InpUseNY = true;                // Trade NY session
input int      InpLondonStart = 8;             // London start (UTC)
input int      InpLondonEnd = 16;              // London end (UTC)
input int      InpNYStart = 13;                // NY start (UTC)
input int      InpNYEnd = 21;                  // NY end (UTC)

input group "=== Magic & Comment ==="
input int      InpMagicNumber = 123456;        // Magic number
input string   InpComment = "QuantEA";         // Order comment

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CTrade         trade;
CPositionInfo  positionInfo;
CAccountInfo   accountInfo;
CSymbolInfo    symbolInfo;

// Risk tracking
double         g_peakBalance;
double         g_dailyStartBalance;
datetime       g_currentDay;
int            g_tradesToday;

// Indicator handles
int            g_atrHandle;
int            g_fastHandle;
int            g_slowHandle;

// Tick buffer for z-score
#define TICK_BUFFER_SIZE 500
double         g_tickPrices[];
int            g_tickIndex;
int            g_tickCount;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    // Validate inputs
    if(InpRiskPercent <= 0 || InpRiskPercent > 5) {
        Print("Error: RiskPercent must be between 0 and 5");
        return INIT_PARAMETERS_INCORRECT;
    }

    if(InpFastPeriod >= InpSlowPeriod) {
        Print("Error: FastPeriod must be less than SlowPeriod");
        return INIT_PARAMETERS_INCORRECT;
    }

    // Initialize symbol info
    if(!symbolInfo.Name(_Symbol)) {
        Print("Error: Failed to initialize symbol info");
        return INIT_FAILED;
    }
    symbolInfo.Refresh();

    // Initialize trade settings
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    // Initialize indicators
    g_atrHandle = iATR(_Symbol, PERIOD_H1, 14);
    g_fastHandle = iMA(_Symbol, PERIOD_M1, InpFastPeriod, 0, MODE_EMA, PRICE_CLOSE);
    g_slowHandle = iMA(_Symbol, PERIOD_M1, InpSlowPeriod, 0, MODE_EMA, PRICE_CLOSE);

    if(g_atrHandle == INVALID_HANDLE ||
       g_fastHandle == INVALID_HANDLE ||
       g_slowHandle == INVALID_HANDLE) {
        Print("Error: Failed to create indicators");
        return INIT_FAILED;
    }

    // Initialize tick buffer
    ArrayResize(g_tickPrices, TICK_BUFFER_SIZE);
    ArrayInitialize(g_tickPrices, 0);
    g_tickIndex = 0;
    g_tickCount = 0;

    // Initialize risk tracking
    g_peakBalance = accountInfo.Balance();
    g_dailyStartBalance = g_peakBalance;
    g_currentDay = TimeCurrent();
    g_tradesToday = 0;

    Print("EA Initialized. Balance: ", g_peakBalance);

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Release indicator handles
    if(g_atrHandle != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
    if(g_fastHandle != INVALID_HANDLE) IndicatorRelease(g_fastHandle);
    if(g_slowHandle != INVALID_HANDLE) IndicatorRelease(g_slowHandle);

    Print("EA Deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    // Daily reset check
    CheckDailyReset();

    // Pre-trade validation
    if(!IsTradeAllowed()) return;
    if(!CheckSpread()) return;
    if(!CheckSession()) return;
    if(!CheckDrawdownLimits()) return;

    // Store tick data
    StoreTick();

    // Check for signals (only if we can open more positions)
    if(CountOpenPositions() < InpMaxPositions) {
        CheckSignals();
    }

    // Manage open positions
    ManagePositions();
}

//+------------------------------------------------------------------+
//| Store tick data for z-score calculation                          |
//+------------------------------------------------------------------+
void StoreTick()
{
    MqlTick tick;
    if(!SymbolInfoTick(_Symbol, tick)) return;

    g_tickPrices[g_tickIndex] = tick.bid;
    g_tickIndex = (g_tickIndex + 1) % TICK_BUFFER_SIZE;
    if(g_tickCount < TICK_BUFFER_SIZE) g_tickCount++;
}

//+------------------------------------------------------------------+
//| Calculate Z-Score from tick data                                  |
//+------------------------------------------------------------------+
double CalculateZScore(int period)
{
    if(g_tickCount < period) return 0;

    // Calculate returns from ticks
    double returns[];
    ArrayResize(returns, period - 1);

    for(int i = 0; i < period - 1; i++) {
        int curr = (g_tickIndex - 1 - i + TICK_BUFFER_SIZE) % TICK_BUFFER_SIZE;
        int prev = (curr - 1 + TICK_BUFFER_SIZE) % TICK_BUFFER_SIZE;

        if(g_tickPrices[prev] > 0) {
            returns[i] = g_tickPrices[curr] - g_tickPrices[prev];
        } else {
            returns[i] = 0;
        }
    }

    // Calculate mean and std (excluding most recent)
    double mean = 0, variance = 0;
    int count = period - 2;

    for(int i = 1; i < period - 1; i++) {
        mean += returns[i];
    }
    mean /= count;

    for(int i = 1; i < period - 1; i++) {
        variance += MathPow(returns[i] - mean, 2);
    }
    double std = MathSqrt(variance / count);

    if(std == 0) return 0;

    // Z-score of most recent return
    return (returns[0] - mean) / std;
}

//+------------------------------------------------------------------+
//| Check for entry signals                                           |
//+------------------------------------------------------------------+
void CheckSignals()
{
    // Get indicator values
    double fast[], slow[], atr[];
    ArraySetAsSeries(fast, true);
    ArraySetAsSeries(slow, true);
    ArraySetAsSeries(atr, true);

    if(CopyBuffer(g_fastHandle, 0, 0, 3, fast) < 3) return;
    if(CopyBuffer(g_slowHandle, 0, 0, 3, slow) < 3) return;
    if(CopyBuffer(g_atrHandle, 0, 0, 1, atr) < 1) return;

    // Calculate z-score
    double zScore = CalculateZScore(InpLookback);

    // Get current prices
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

    // Calculate SL/TP distances
    double slDistance = atr[0] * InpSLMultiplier;
    double tpDistance = atr[0] * InpTPMultiplier;

    // Entry conditions
    // BUY: Fast > Slow AND z-score indicates oversold (negative)
    bool buyCondition = (fast[1] > slow[1]) &&
                        (fast[2] <= slow[2]) &&
                        (zScore < -InpZScoreEntry);

    // SELL: Fast < Slow AND z-score indicates overbought (positive)
    bool sellCondition = (fast[1] < slow[1]) &&
                         (fast[2] >= slow[2]) &&
                         (zScore > InpZScoreEntry);

    // Execute trades
    if(buyCondition) {
        double sl = NormalizeDouble(ask - slDistance, _Digits);
        double tp = NormalizeDouble(ask + tpDistance, _Digits);
        double lots = CalculateLotSize(slDistance);

        if(lots > 0) {
            OpenBuy(lots, sl, tp);
        }
    }
    else if(sellCondition) {
        double sl = NormalizeDouble(bid + slDistance, _Digits);
        double tp = NormalizeDouble(bid - tpDistance, _Digits);
        double lots = CalculateLotSize(slDistance);

        if(lots > 0) {
            OpenSell(lots, sl, tp);
        }
    }
}

//+------------------------------------------------------------------+
//| Calculate position size based on risk                             |
//+------------------------------------------------------------------+
double CalculateLotSize(double slDistance)
{
    double balance = accountInfo.Balance();
    double riskAmount = balance * InpRiskPercent / 100.0;

    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

    if(tickSize == 0) return 0;

    double pointValue = tickValue / tickSize * _Point;
    double slPoints = slDistance / _Point;

    double lots = riskAmount / (slPoints * pointValue);

    // Normalize to lot constraints
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lots = MathFloor(lots / lotStep) * lotStep;
    lots = MathMax(minLot, MathMin(maxLot, lots));

    return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| Open buy position                                                 |
//+------------------------------------------------------------------+
bool OpenBuy(double lots, double sl, double tp)
{
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

    if(trade.Buy(lots, _Symbol, ask, sl, tp, InpComment)) {
        g_tradesToday++;
        Print("BUY opened: ", lots, " lots at ", ask, " SL:", sl, " TP:", tp);
        return true;
    }

    Print("BUY failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
    return false;
}

//+------------------------------------------------------------------+
//| Open sell position                                                |
//+------------------------------------------------------------------+
bool OpenSell(double lots, double sl, double tp)
{
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    if(trade.Sell(lots, _Symbol, bid, sl, tp, InpComment)) {
        g_tradesToday++;
        Print("SELL opened: ", lots, " lots at ", bid, " SL:", sl, " TP:", tp);
        return true;
    }

    Print("SELL failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
    return false;
}

//+------------------------------------------------------------------+
//| Manage open positions (trailing stop)                             |
//+------------------------------------------------------------------+
void ManagePositions()
{
    if(!InpUseTrailing) return;

    double atr[];
    ArraySetAsSeries(atr, true);
    if(CopyBuffer(g_atrHandle, 0, 0, 1, atr) < 1) return;

    double trailDistance = atr[0] * InpTrailMultiplier;

    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        if(positionInfo.SelectByIndex(i)) {
            if(positionInfo.Symbol() != _Symbol) continue;
            if(positionInfo.Magic() != InpMagicNumber) continue;

            ulong ticket = positionInfo.Ticket();
            double openPrice = positionInfo.PriceOpen();
            double currentSL = positionInfo.StopLoss();
            ENUM_POSITION_TYPE posType = positionInfo.PositionType();

            if(posType == POSITION_TYPE_BUY) {
                double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                double newSL = NormalizeDouble(bid - trailDistance, _Digits);

                if(newSL > currentSL && newSL > openPrice) {
                    trade.PositionModify(ticket, newSL, positionInfo.TakeProfit());
                }
            }
            else {
                double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                double newSL = NormalizeDouble(ask + trailDistance, _Digits);

                if((currentSL == 0 || newSL < currentSL) && newSL < openPrice) {
                    trade.PositionModify(ticket, newSL, positionInfo.TakeProfit());
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Count open positions for this EA                                  |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        if(positionInfo.SelectByIndex(i)) {
            if(positionInfo.Symbol() == _Symbol &&
               positionInfo.Magic() == InpMagicNumber) {
                count++;
            }
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                       |
//+------------------------------------------------------------------+
bool IsTradeAllowed()
{
    if(!MQLInfoInteger(MQL_TRADE_ALLOWED)) return false;
    if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)) return false;
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) return false;

    ENUM_SYMBOL_TRADE_MODE mode =
        (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE);

    return (mode == SYMBOL_TRADE_MODE_FULL);
}

//+------------------------------------------------------------------+
//| Check spread filter                                               |
//+------------------------------------------------------------------+
bool CheckSpread()
{
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double spread = (ask - bid) / _Point;

    return (spread <= InpMaxSpread);
}

//+------------------------------------------------------------------+
//| Check session filter                                              |
//+------------------------------------------------------------------+
bool CheckSession()
{
    if(!InpUseLondon && !InpUseNY) return true;

    MqlDateTime now;
    TimeToStruct(TimeGMT(), now);
    int hour = now.hour;

    bool inLondon = InpUseLondon && (hour >= InpLondonStart && hour < InpLondonEnd);
    bool inNY = InpUseNY && (hour >= InpNYStart && hour < InpNYEnd);

    return (inLondon || inNY);
}

//+------------------------------------------------------------------+
//| Check drawdown limits                                             |
//+------------------------------------------------------------------+
bool CheckDrawdownLimits()
{
    double balance = accountInfo.Balance();
    double equity = accountInfo.Equity();

    // Update peak balance
    if(balance > g_peakBalance) g_peakBalance = balance;

    // Daily drawdown
    double dailyDD = (g_dailyStartBalance - equity) / g_dailyStartBalance * 100;
    if(dailyDD >= InpMaxDailyDD) {
        Print("Daily drawdown limit reached: ", DoubleToString(dailyDD, 2), "%");
        CloseAllPositions();
        return false;
    }

    // Total drawdown
    double totalDD = (g_peakBalance - equity) / g_peakBalance * 100;
    if(totalDD >= InpMaxTotalDD) {
        Print("Total drawdown limit reached: ", DoubleToString(totalDD, 2), "%");
        CloseAllPositions();
        return false;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Check for daily reset                                             |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
    MqlDateTime now, last;
    TimeToStruct(TimeCurrent(), now);
    TimeToStruct(g_currentDay, last);

    if(now.day != last.day) {
        g_dailyStartBalance = accountInfo.Balance();
        g_currentDay = TimeCurrent();
        g_tradesToday = 0;
        Print("New trading day. Balance: ", g_dailyStartBalance);
    }
}

//+------------------------------------------------------------------+
//| Close all positions                                               |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        if(positionInfo.SelectByIndex(i)) {
            if(positionInfo.Symbol() == _Symbol &&
               positionInfo.Magic() == InpMagicNumber) {
                trade.PositionClose(positionInfo.Ticket());
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Custom tester function for optimization                           |
//+------------------------------------------------------------------+
double OnTester()
{
    double profit = TesterStatistics(STAT_PROFIT);
    double trades = TesterStatistics(STAT_TRADES);
    double drawdown = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
    double sharpe = TesterStatistics(STAT_SHARPE_RATIO);
    double profitFactor = TesterStatistics(STAT_PROFIT_FACTOR);

    // Minimum trade requirement
    if(trades < 100) return 0;

    // Custom fitness: Sharpe * PF / (1 + DD/10)
    double fitness = sharpe * profitFactor / (1 + drawdown / 10);

    return fitness;
}
//+------------------------------------------------------------------+
