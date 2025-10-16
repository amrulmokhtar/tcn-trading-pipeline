using System;
using System.Linq;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;
using cAlgo.API.Indicators;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SMC_2_6_1_VolumeFlow_OBV_Adaptive : Robot
    {
        // ===== General =====
        [Parameter("Symbol", Group = "General", DefaultValue = "XAUUSD")]
        public string TradeSymbol { get; set; }

        // ===== Guards =====
        [Parameter("Max Loss Cap, pips", Group = "Guards", DefaultValue = 1500, MinValue = 100, MaxValue = 200000)]
        public int MaxLossPips { get; set; }

        [Parameter("Max Spread, points", Group = "Guards", DefaultValue = 25, MinValue = 1, MaxValue = 500)]
        public int MaxSpreadPoints { get; set; }

        [Parameter("Use Soft Spread Block", Group = "Guards", DefaultValue = true)]
        public bool UseSoftSpreadBlock { get; set; }

        [Parameter("Soft Spread Block frac", Group = "Guards", DefaultValue = 0.6, MinValue = 0.3, MaxValue = 1.0)]
        public double SoftSpreadBlockFrac { get; set; }

        [Parameter("Spread/ATR cap", Group = "Guards", DefaultValue = 0.05, MinValue = 0.01, MaxValue = 0.20, Step = 0.005)]
        public double SpreadToAtrCap { get; set; }

        // ===== Risk =====
        [Parameter("Use Risk %", Group = "Risk", DefaultValue = true)]
        public bool UseRiskPercent { get; set; }

        [Parameter("Risk % per trade", Group = "Risk", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 5.0)]
        public double RiskPercent { get; set; }

        [Parameter("Fixed Volume, lots", Group = "Risk", DefaultValue = 0.02, MinValue = 0.01, MaxValue = 10)]
        public double FixedVolumeLots { get; set; }

        // ===== H1 Regime (kept) =====
        [Parameter("Use Regime Filter", Group = "H1 Regime", DefaultValue = true)]
        public bool UseRegimeFilter { get; set; }

        [Parameter("H1 Fast SMA", Group = "H1 Regime", DefaultValue = 20, MinValue = 5, MaxValue = 100)]
        public int H1FastSMA { get; set; }

        [Parameter("H1 Slow SMA", Group = "H1 Regime", DefaultValue = 200, MinValue = 50, MaxValue = 400)]
        public int H1SlowSMA { get; set; }

        [Parameter("Use Range Skip", Group = "H1 Regime", DefaultValue = true)]
        public bool UseRangeSkip { get; set; }

        [Parameter("Range Sep ATR Long", Group = "H1 Regime", DefaultValue = 0.35, MinValue = 0.0, MaxValue = 2.0)]
        public double RangeSepATR_Long { get; set; }

        [Parameter("Range Sep ATR Short", Group = "H1 Regime", DefaultValue = 0.45, MinValue = 0.0, MaxValue = 2.0)]
        public double RangeSepATR_Short { get; set; }

        // ===== ADX strength gate on H1 (kept) =====
        [Parameter("Use H1 ADX Filter", Group = "ADX", DefaultValue = true)]
        public bool UseADXFilter { get; set; }

        [Parameter("ADX Period", Group = "ADX", DefaultValue = 14, MinValue = 5, MaxValue = 50)]
        public int ADXPeriod { get; set; }

        [Parameter("ADX Long Min", Group = "ADX", DefaultValue = 22, MinValue = 5, MaxValue = 50)]
        public double ADXLongMin { get; set; }

        [Parameter("ADX Short Min", Group = "ADX", DefaultValue = 30, MinValue = 5, MaxValue = 50)]
        public double ADXShortMin { get; set; }

        // ===== Volume + Flow quality gates =====
        [Parameter("Use H1 Volume Percentile", Group = "Volume", DefaultValue = true)]
        public bool UseH1VolumePct { get; set; }

        [Parameter("H1 Volume Lookback", Group = "Volume", DefaultValue = 120, MinValue = 50, MaxValue = 500)]
        public int H1VolumeLookback { get; set; }

        [Parameter("H1 Volume Pct Min", Group = "Volume", DefaultValue = 35, MinValue = 0, MaxValue = 100)]
        public int H1VolumePctMin { get; set; }

        [Parameter("H1 Volume Pct Max", Group = "Volume", DefaultValue = 90, MinValue = 0, MaxValue = 100)]
        public int H1VolumePctMax { get; set; }

        [Parameter("Use H1 OBV Confirm", Group = "OBV", DefaultValue = true)]
        public bool UseH1ObvConfirm { get; set; }

        [Parameter("H1 OBV SMA", Group = "OBV", DefaultValue = 50, MinValue = 10, MaxValue = 200)]
        public int H1ObvSma { get; set; }

        [Parameter("H1 OBV Slope Bars", Group = "OBV", DefaultValue = 6, MinValue = 3, MaxValue = 24)]
        public int H1ObvSlopeBars { get; set; }

        // ===== Session filter Malaysia default =====
        [Parameter("Use Session Filter", Group = "Session", DefaultValue = true)]
        public bool UseSessionFilter { get; set; }

        [Parameter("Session TZ Offset (h)", Group = "Session", DefaultValue = 8, MinValue = -12, MaxValue = 14)]
        public int SessionTzOffsetHours { get; set; }

        [Parameter("Session Start Hour", Group = "Session", DefaultValue = 9, MinValue = 0, MaxValue = 23)]
        public int SessionStartHour { get; set; }

        [Parameter("Session End Hour", Group = "Session", DefaultValue = 3, MinValue = 0, MaxValue = 23)]
        public int SessionEndHour { get; set; }

        [Parameter("Short Blackout From Start (h)", Group = "Session", DefaultValue = 2, MinValue = 0, MaxValue = 12)]
        public int ShortBlackoutFromStart { get; set; }

        [Parameter("Short Blackout Before End (h)", Group = "Session", DefaultValue = 1, MinValue = 0, MaxValue = 12)]
        public int ShortBlackoutBeforeEnd { get; set; }

        [Parameter("Block Longs Start Hour", Group = "Session", DefaultValue = 21, MinValue = 0, MaxValue = 23)]
        public int LongBlockStartHour { get; set; }

        [Parameter("Block Longs End Hour", Group = "Session", DefaultValue = 24, MinValue = 0, MaxValue = 24)]
        public int LongBlockEndHour { get; set; }

        // ===== Entry controls per side =====
        [Parameter("Max Open Positions", Group = "Entry Controls", DefaultValue = 1, MinValue = 1, MaxValue = 10)]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min Bars Between Longs", Group = "Entry Controls", DefaultValue = 18, MinValue = 0, MaxValue = 200)]
        public int MinBarsBetweenLongs { get; set; }

        [Parameter("Min Bars Between Shorts", Group = "Entry Controls", DefaultValue = 20, MinValue = 0, MaxValue = 200)]
        public int MinBarsBetweenShorts { get; set; }

        [Parameter("Min Entry Distance ATR Long", Group = "Entry Controls", DefaultValue = 0.7, MinValue = 0.0, MaxValue = 5.0)]
        public double MinEntryDistanceATR_Long { get; set; }

        [Parameter("Min Entry Distance ATR Short", Group = "Entry Controls", DefaultValue = 0.8, MinValue = 0.0, MaxValue = 5.0)]
        public double MinEntryDistanceATR_Short { get; set; }

        [Parameter("Min Pullback ATR Long", Group = "Entry Controls", DefaultValue = 0.5, MinValue = 0.0, MaxValue = 5.0)]
        public double MinPullbackATR_Long { get; set; }

        [Parameter("Min Pullback ATR Short", Group = "Entry Controls", DefaultValue = 0.7, MinValue = 0.0, MaxValue = 5.0)]
        public double MinPullbackATR_Short { get; set; }

        // ===== Management per side =====
        [Parameter("BE at R Long", Group = "Mgmt Long", DefaultValue = 1.25)]
        public double BreakEvenR_Long { get; set; }

        [Parameter("BE at R Short", Group = "Mgmt Short", DefaultValue = 1.35)]
        public double BreakEvenR_Short { get; set; }

        [Parameter("BE ATR buffer frac", Group = "Management", DefaultValue = 0.12, MinValue = 0.0, MaxValue = 0.5)]
        public double BreakEvenATRBufferFrac { get; set; }

        [Parameter("Partial1 at R", Group = "Management", DefaultValue = 1.5)]
        public double Partial1R { get; set; }

        [Parameter("Partial1 percent", Group = "Management", DefaultValue = 25, MinValue = 10, MaxValue = 90)]
        public double Partial1Percent { get; set; }

        [Parameter("Trail start R Long", Group = "Mgmt Long", DefaultValue = 2.2)]
        public double ATRTrailStartR_Long { get; set; }

        [Parameter("Trail start R Short", Group = "Mgmt Short", DefaultValue = 2.4)]
        public double ATRTrailStartR_Short { get; set; }

        [Parameter("ATR Mult Long", Group = "Mgmt Long", DefaultValue = 1.6)]
        public double ATRMult_Long { get; set; }

        [Parameter("ATR Mult Short", Group = "Mgmt Short", DefaultValue = 1.8)]
        public double ATRMult_Short { get; set; }

        [Parameter("Max Hold Bars", Group = "Management", DefaultValue = 240, MinValue = 10, MaxValue = 10000)]
        public int MaxHoldBars { get; set; }

        // ===== Internals =====
        private AverageTrueRange _atr;
        private SimpleMovingAverage _maFastM15;
        private HashSet<long> _beSet = new HashSet<long>();
        private HashSet<long> _partial1Done = new HashSet<long>();
        private int _lastLongBar = -1;
        private int _lastShortBar = -1;

        // Adaptive controls, no new public parameters
        private const int ADX_LOOKBACK = 50;            // for adaptive thresholding
        private const int ATR_LOOKBACK = 200;           // for ATR percentile regime
        private const double ADX_STD_WEIGHT = 0.5;      // threshold = mean + w * std
        private const int VOL_PCT_HIGH_ATR_BUMP = 10;   // increase min by 10 when ATR high
        private const int VOL_PCT_HIGH_ATR_TRIM = 10;   // decrease max by 10 when ATR high
        private const int HIGH_ATR_PCTL = 80;           // high volatility if ATR percentile >= 80
        private const int LOW_ATR_PCTL  = 20;           // low volatility if ATR percentile <= 20

        protected override void OnStart()
        {
            _atr = Indicators.AverageTrueRange(Bars, 14, MovingAverageType.Exponential);
            _maFastM15 = Indicators.SimpleMovingAverage(Bars.ClosePrices, 20);
        }

        protected override void OnTick()
        {
            EnforcePerPositionMaxLossPips();
        }

        protected override void OnBar()
        {
            EnforcePerPositionMaxLossPips();
            ManageOpenPositions();

            if (Positions.Count(p => p.SymbolName == SymbolName) >= MaxOpenPositions) return;
            if (!InSessionMalaysia(Server.Time)) return;
            if (!SpreadOk()) return;
            if (!SpreadVsAtrOk()) return;

            // H1 regime and range skip gives base direction
            int dir = TrendDirectionWithRange();
            if (UseRegimeFilter && dir == 0) return;

            // Compute adaptive stats on H1
            int atrPct = H1ATRPercentile(ATR_LOOKBACK, 14);
            var adxStats = H1ADX_MeanStd(ADXPeriod, ADX_LOOKBACK);
            double adxLongDyn = UseADXFilter ? Math.Max(5.0, Math.Max(ADXLongMin, adxStats.mean + ADX_STD_WEIGHT * adxStats.std)) : 0;
            double adxShortDyn = UseADXFilter ? Math.Max(5.0, Math.Max(ADXShortMin, adxStats.mean + ADX_STD_WEIGHT * adxStats.std)) : 0;

            // ADX strength gate on H1 with adaptive thresholds
            double h1Adx = UseADXFilter ? H1ADX(ADXPeriod) : 999;
            if (UseADXFilter)
            {
                if (h1Adx < 0) return;
                if (dir > 0 && h1Adx < adxLongDyn) return;
                if (dir < 0 && h1Adx < adxShortDyn) return;
            }

            // H1 Volume percentile with adaptive bounds based on ATR regime
            if (UseH1VolumePct)
            {
                int volMin = H1VolumePctMin;
                int volMax = H1VolumePctMax;

                if (atrPct >= HIGH_ATR_PCTL)
                {
                    volMin = ClampInt(volMin + VOL_PCT_HIGH_ATR_BUMP, 0, 100);
                    volMax = ClampInt(volMax - VOL_PCT_HIGH_ATR_TRIM, 0, 100);
                    if (volMin > volMax) volMin = Math.Max(0, volMax - 5);
                }
                else if (atrPct <= LOW_ATR_PCTL)
                {
                    // In quieter regimes, widen the window a bit
                    volMin = ClampInt(volMin - 5, 0, 100);
                    volMax = ClampInt(volMax + 5, 0, 100);
                }

                double pct = H1VolumePercentile();
                if (pct < 0) return;
                if (pct < volMin || pct > volMax) return;
            }

            // H1 OBV flow confirm
            if (UseH1ObvConfirm)
            {
                var obvState = H1ObvState(H1ObvSma, H1ObvSlopeBars);
                if (dir > 0 && !obvState.LongOk) return;
                if (dir < 0 && !obvState.ShortOk) return;
            }

            if (Bars.Count < 50) return;

            var sym = Symbols.GetSymbol(SymbolName);
            double atrNow = _atr.Result.LastValue;
            if (atrNow <= 0) return;

            double c1 = Bars.ClosePrices.Last(1);
            double o1 = Bars.OpenPrices.Last(1);
            double c2 = Bars.ClosePrices.Last(2);
            double o2 = Bars.OpenPrices.Last(2);

            bool bullReversal = c1 > o1 && c2 < o2;
            bool bearReversal = c1 < o1 && c2 > o2;

            // M15 alignment
            double ma = _maFastM15.Result.LastValue;
            bool longAligned = c1 > ma;
            bool shortAligned = c1 < ma;

            // Pullback distance to MA in ATR units
            double pullbackATR = Math.Abs(c1 - ma) / Math.Max(1e-8, atrNow);

            // Min distance from last entry in same direction
            double distFromLastLongATR = GetDistanceFromLastEntryATR(TradeType.Buy, atrNow);
            double distFromLastShortATR = GetDistanceFromLastEntryATR(TradeType.Sell, atrNow);

            if (dir >= 1 && bullReversal && longAligned && !LongBlocked(Server.Time))
            {
                if (Bars.Count - _lastLongBar < MinBarsBetweenLongs) return;
                if (pullbackATR < MinPullbackATR_Long) return;
                if (!double.IsNaN(distFromLastLongATR) && distFromLastLongATR < MinEntryDistanceATR_Long) return;

                double lots = UseRiskPercent ? CalcRiskLots(Math.Max(2, atrNow / sym.PipSize)) : FixedVolumeLots;
                ExecuteMarketOrder(TradeType.Buy, SymbolName, Symbol.QuantityToVolumeInUnits(lots));
                _lastLongBar = Bars.Count;
            }
            else if (dir <= -1 && bearReversal && shortAligned && !ShortBlackoutActive(Server.Time))
            {
                if (Bars.Count - _lastShortBar < MinBarsBetweenShorts) return;
                if (pullbackATR < MinPullbackATR_Short) return;
                if (!double.IsNaN(distFromLastShortATR) && distFromLastShortATR < MinEntryDistanceATR_Short) return;

                double lots = UseRiskPercent ? CalcRiskLots(Math.Max(2, atrNow / sym.PipSize)) : FixedVolumeLots;
                ExecuteMarketOrder(TradeType.Sell, SymbolName, Symbol.QuantityToVolumeInUnits(lots));
                _lastShortBar = Bars.Count;
            }
        }

        // ===== Management =====
        private void ManageOpenPositions()
        {
            foreach (var p in Positions.Where(p => p.SymbolName == SymbolName).ToArray())
            {
                int barsAlive = Bars.OpenTimes.Count(t => t >= p.EntryTime);
                if (barsAlive >= MaxHoldBars)
                {
                    ClosePosition(p);
                    continue;
                }

                double slPips = PipsFromPrice(p.EntryPrice, p.StopLoss ?? p.EntryPrice);
                double rNow = 0.0;
                if (slPips > 0)
                {
                    var sym = Symbols.GetSymbol(p.SymbolName);
                    double distance = p.TradeType == TradeType.Buy ? sym.Bid - p.EntryPrice : p.EntryPrice - sym.Ask;
                    rNow = (distance / sym.PipSize) / slPips;
                }

                double beR = p.TradeType == TradeType.Buy ? BreakEvenR_Long : BreakEvenR_Short;
                if (beR > 0 && rNow >= beR && !_beSet.Contains(p.Id))
                {
                    var symBE = Symbols.GetSymbol(p.SymbolName);
                    double spreadPips = Math.Max(0.0, (symBE.Ask - symBE.Bid) / symBE.PipSize);
                    double atrPips    = _atr != null ? (_atr.Result.LastValue / symBE.PipSize) : 0.0;
                    double bufferPips = spreadPips + Math.Max(1.0, BreakEvenATRBufferFrac * atrPips);
                    double be = p.EntryPrice + (p.TradeType == TradeType.Buy ? 1.0 : -1.0) * bufferPips * symBE.PipSize;
                    ModifyPosition(p, be, p.TakeProfit, null, false);
                    _beSet.Add(p.Id);
                }

                if (Partial1R > 0 && rNow >= Partial1R && !_partial1Done.Contains(p.Id))
                {
                    double closeUnits = p.VolumeInUnits * (Partial1Percent / 100.0);
                    try
                    {
                        long u = (long)Math.Round(closeUnits);
                        if (u >= p.VolumeInUnits) ClosePosition(p);
                        else ClosePosition(p, u);
                        _partial1Done.Add(p.Id);
                    } catch {}
                }

                if (p.TradeType == TradeType.Buy)
                {
                    if (rNow >= ATRTrailStartR_Long)
                    {
                        var symT = Symbols.GetSymbol(p.SymbolName);
                        double trail = _atr.Result.LastValue * ATRMult_Long;
                        double newSL = Math.Max(p.StopLoss ?? double.MinValue, symT.Bid - trail);
                        if (p.StopLoss == null || newSL > p.StopLoss) ModifyPosition(p, newSL, p.TakeProfit, null, false);
                    }
                }
                else
                {
                    if (rNow >= ATRTrailStartR_Short)
                    {
                        var symT = Symbols.GetSymbol(p.SymbolName);
                        double trail = _atr.Result.LastValue * ATRMult_Short;
                        double newSL = Math.Min(p.StopLoss ?? double.MaxValue, symT.Ask + trail);
                        if (p.StopLoss == null || newSL < p.StopLoss) ModifyPosition(p, newSL, p.TakeProfit, null, false);
                    }
                }
            }
        }

        // ===== Safety, per position max loss pips =====
        private void EnforcePerPositionMaxLossPips()
        {
            if (MaxLossPips <= 0) return;

            foreach (var p in Positions.ToArray())
            {
                if (p.SymbolName != SymbolName && p.SymbolName != TradeSymbol) continue;
                var sym = Symbols.GetSymbol(p.SymbolName);
                if (sym == null) continue;

                double current = p.TradeType == TradeType.Buy ? sym.Bid : sym.Ask;
                double lossPips = p.TradeType == TradeType.Buy
                    ? (p.EntryPrice - current) / sym.PipSize
                    : (current - p.EntryPrice) / sym.PipSize;

                if (lossPips > MaxLossPips)
                {
                    Print("Force close {0} exceeded max loss {1} pips, loss now {2:F0} pips", p.Id, MaxLossPips, lossPips);
                    try { ClosePosition(p); } catch {}
                }
            }
        }

        // ===== Regime and hygiene helpers =====
        private bool InSessionMalaysia(DateTime serverTime)
        {
            if (!UseSessionFilter) return true;
            var local = serverTime.AddHours(SessionTzOffsetHours);
            int s = SessionStartHour, e = SessionEndHour;
            if (s == e) return true;
            if (s <= e) return local.Hour >= s && local.Hour < e;
            return local.Hour >= s || local.Hour < e;
        }

        private bool LongBlocked(DateTime serverTime)
        {
            if (!UseSessionFilter) return false;
            var local = serverTime.AddHours(SessionTzOffsetHours);
            int start = LongBlockStartHour;
            int end = LongBlockEndHour % 24;
            if (start == end) return true;
            if (start < end) return local.Hour >= start && local.Hour < end;
            return local.Hour >= start || local.Hour < end;
        }

        private bool ShortBlackoutActive(DateTime serverTime)
        {
            if (!UseSessionFilter) return false;
            var local = serverTime.AddHours(SessionTzOffsetHours);
            int s = SessionStartHour, e = SessionEndHour;
            int hoursSinceStart, hoursToEnd;
            if (s <= e)
            {
                hoursSinceStart = (local.Hour - s + 24) % 24;
                hoursToEnd = (e - local.Hour + 24) % 24;
            }
            else
            {
                bool inFirst = local.Hour >= s;
                hoursSinceStart = inFirst ? (local.Hour - s) : (local.Hour + (24 - s));
                bool beforeEnd = local.Hour < e;
                hoursToEnd = beforeEnd ? (e - local.Hour) : (24 - (local.Hour - e));
            }
            if (hoursSinceStart < ShortBlackoutFromStart) return true;
            if (hoursToEnd <= ShortBlackoutBeforeEnd && hoursToEnd >= 0) return true;
            return false;
        }

        private bool SpreadOk()
        {
            var sym = Symbols.GetSymbol(TradeSymbol);
            double points = (sym.Ask - sym.Bid) / sym.PipSize;
            if (points > MaxSpreadPoints) return false;
            if (UseSoftSpreadBlock)
            {
                double softCap = SoftSpreadBlockFrac * MaxSpreadPoints;
                if (points > softCap) return false;
            }
            return true;
        }

        private bool SpreadVsAtrOk()
        {
            if (_atr == null) return true;
            var sym = Symbols.GetSymbol(TradeSymbol);
            double spreadPips = (sym.Ask - sym.Bid) / sym.PipSize;
            double atrPips = _atr.Result.LastValue / sym.PipSize;
            if (atrPips <= 0) return true;
            double ratio = spreadPips / atrPips;
            return ratio <= SpreadToAtrCap;
        }

        private bool TryGetH1Bars(out Bars h1)
        {
            h1 = MarketData.GetBars(TimeFrame.Hour, TradeSymbol);
            return h1 != null && h1.Count > Math.Max(H1SlowSMA, Math.Max(H1VolumeLookback, 50));
        }

        private double SMA(Bars b, int period)
        {
            double sum = 0.0;
            for (int i = 1; i <= period; i++) sum += b.ClosePrices.Last(i);
            return sum / period;
        }

        private double ATR_Wilder(Bars b, int period = 14)
        {
            double atr = 0.0;
            for (int i = period; i >= 1; i--)
            {
                double high = b.HighPrices.Last(i);
                double low  = b.LowPrices.Last(i);
                double prevClose = b.ClosePrices.Last(i + 1);
                double tr = Math.Max(high - low, Math.Max(Math.Abs(high - prevClose), Math.Abs(low - prevClose)));
                atr += tr;
            }
            return atr / period;
        }

        // Approximate ADX on H1 using Wilder init over last "period" bars
        private double H1ADX(int period)
        {
            return H1ADXAt(period, 0);
        }

        // ADX computed ending at offset bars ago
        private double H1ADXAt(int period, int offset)
        {
            if (!TryGetH1Bars(out var h1)) return -1;
            if (h1.Count < period + 2 + offset) return -1;

            double prevHigh = h1.HighPrices.Last(period + 1 + offset);
            double prevLow  = h1.LowPrices.Last(period + 1 + offset);
            double prevClose= h1.ClosePrices.Last(period + 1 + offset);

            double sumTR = 0.0, sumPlusDM = 0.0, sumMinusDM = 0.0;

            for (int i = period + offset; i >= 1 + offset; i--)
            {
                double high = h1.HighPrices.Last(i);
                double low  = h1.LowPrices.Last(i);
                double upMove = high - prevHigh;
                double downMove = prevLow - low;

                double tr = Math.Max(high - low, Math.Max(Math.Abs(high - prevClose), Math.Abs(low - prevClose)));
                sumTR += tr;
                sumPlusDM  += (upMove > downMove && upMove > 0) ? upMove : 0.0;
                sumMinusDM += (downMove > upMove && downMove > 0) ? downMove : 0.0;

                prevHigh = high;
                prevLow  = low;
                prevClose = h1.ClosePrices.Last(i);
            }

            if (sumTR <= 0) return -1;

            double diPlus  = 100.0 * (sumPlusDM / sumTR);
            double diMinus = 100.0 * (sumMinusDM / sumTR);
            double denom = diPlus + diMinus;
            if (denom <= 0) return -1;
            double dx = 100.0 * Math.Abs(diPlus - diMinus) / denom;

            return dx; // approximation of ADX
        }

        // Mean and Std of ADX over a lookback window
        private (double mean, double std) H1ADX_MeanStd(int period, int lookback)
        {
            List<double> vals = new List<double>();
            for (int k = 0; k < lookback; k++)
            {
                double v = H1ADXAt(period, k);
                if (v >= 0) vals.Add(v);
            }
            if (vals.Count == 0) return (0, 0);
            double m = vals.Average();
            double s = Math.Sqrt(vals.Select(x => (x - m) * (x - m)).DefaultIfEmpty(0).Average());
            return (m, s);
        }

        // ATR percentile on H1
        private int H1ATRPercentile(int lookback, int period)
        {
            if (!TryGetH1Bars(out var h1)) return -1;
            int n = Math.Min(lookback, h1.Count - period - 2);
            if (n <= 10) return -1;

            // current ATR
            double atrNow = ATR_Wilder(h1, period);

            int rank = 0;
            for (int i = period + 2; i <= n + period + 1; i++)
            {
                double atrPast = 0.0;
                // compute ATR ending at i bars ago
                double prevClose = h1.ClosePrices.Last(i + 1);
                double sum = 0.0;
                for (int j = i; j > i - period; j--)
                {
                    double high = h1.HighPrices.Last(j);
                    double low  = h1.LowPrices.Last(j);
                    double tr = Math.Max(high - low, Math.Max(Math.Abs(high - prevClose), Math.Abs(low - prevClose)));
                    sum += tr;
                    prevClose = h1.ClosePrices.Last(j);
                }
                atrPast = sum / period;
                if (atrPast <= atrNow) rank++;
            }
            double pct = 100.0 * rank / n;
            return (int)Math.Round(pct);
        }

        // Trend direction with per side range skip on H1
        private int TrendDirectionWithRange()
        {
            if (!UseRegimeFilter) return 0;
            if (!TryGetH1Bars(out var h1)) return 0;

            int nFast = Math.Max(5, H1FastSMA);
            int nSlow = Math.Max(nFast + 1, H1SlowSMA);
            double fast = SMA(h1, nFast);
            double slow = SMA(h1, nSlow);

            int dir = fast > slow ? 1 : fast < slow ? -1 : 0;

            if (UseRangeSkip && dir != 0)
            {
                double atr = ATR_Wilder(h1, 14);
                if (atr > 0)
                {
                    double sep = Math.Abs(fast - slow);
                    double ratio = sep / atr;
                    if (dir > 0 && ratio < RangeSepATR_Long) return 0;
                    if (dir < 0 && ratio < RangeSepATR_Short) return 0;
                }
            }
            return dir;
        }

        // ===== H1 volume percentile =====
        private double H1VolumePercentile()
        {
            if (!TryGetH1Bars(out var h1)) return -1;
            int n = Math.Min(H1VolumeLookback, h1.Count - 2);
            if (n <= 10) return -1;
            double vNow = h1.TickVolumes.Last(1);
            int rank = 0;
            for (int i = 2; i <= n + 1; i++)
            {
                if (h1.TickVolumes.Last(i) <= vNow) rank++;
            }
            double pct = 100.0 * rank / n;
            return pct;
        }

        // ===== H1 OBV confirm =====
        private struct ObvState
        {
            public bool LongOk;
            public bool ShortOk;
        }

        private ObvState H1ObvState(int smaLen, int slopeBars)
        {
            var state = new ObvState { LongOk = false, ShortOk = false };
            if (!TryGetH1Bars(out var h1)) return state;
            if (h1.Count < Math.Max(smaLen + slopeBars + 5, 60)) return state;

            // Build OBV cumulative up to the current bar
            double obv = 0.0;
            int total = Math.Min(h1.Count - 1, smaLen + slopeBars + 60);
            for (int i = total; i >= 1; i--)
            {
                double c = h1.ClosePrices.Last(i);
                double prev = h1.ClosePrices.Last(i + 1);
                double vol = h1.TickVolumes.Last(i);
                if (c > prev) obv += vol;
                else if (c < prev) obv -= vol;
            }

            double obvNow = obv;
            double obvPast = obv;
            for (int i = slopeBars; i >= 1; i--)
            {
                double c = h1.ClosePrices.Last(i + 1);
                double prev = h1.ClosePrices.Last(i + 2);
                double vol = h1.TickVolumes.Last(i + 1);
                if (c > prev) obvPast -= vol;
                else if (c < prev) obvPast += vol;
            }

            double obvSma = obvNow;
            for (int i = smaLen; i >= 1; i--)
            {
                double c = h1.ClosePrices.Last(i + 1);
                double prev = h1.ClosePrices.Last(i + 2);
                double vol = h1.TickVolumes.Last(i + 1);
                if (c > prev) obvSma -= vol;
                else if (c < prev) obvSma += vol;
            }
            obvSma = (obvNow + obvSma) / 2.0;

            bool obvAbove = obvNow >= obvSma;
            bool obvBelow = obvNow <= obvSma;
            bool slopeUp = obvNow - obvPast > 0;
            bool slopeDn = obvNow - obvPast < 0;

            state.LongOk = obvAbove && slopeUp;
            state.ShortOk = obvBelow && slopeDn;
            return state;
        }

        private double GetDistanceFromLastEntryATR(TradeType dir, double atrNow)
        {
            var last = Positions.Where(p => p.SymbolName == SymbolName && p.TradeType == dir)
                                .OrderByDescending(p => p.EntryTime).FirstOrDefault();
            if (last == null) return double.NaN;
            var sym = Symbols.GetSymbol(SymbolName);
            double price = dir == TradeType.Buy ? sym.Bid : sym.Ask;
            double dist = Math.Abs(price - last.EntryPrice);
            return dist / Math.Max(1e-8, atrNow);
        }

        private double PipsFromPrice(double a, double b)
        {
            var sym = Symbols.GetSymbol(SymbolName);
            return Math.Abs((a - b) / sym.PipSize);
        }

        private double CalcRiskLots(double slPips)
        {
            var sym = Symbols.GetSymbol(SymbolName);
            double riskUsd = Account.Equity * (RiskPercent / 100.0);
            double pipValuePerLot = sym.PipValue * 100000.0;
            double lots = riskUsd / Math.Max(1.0, slPips * pipValuePerLot);
            return Math.Max(0.01, lots);
        }

        private int ClampInt(int v, int lo, int hi)
        {
            if (v < lo) return lo;
            if (v > hi) return hi;
            return v;
        }
    }
}