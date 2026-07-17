import sqlite3
from collections import namedtuple
import datetime
import numpy as np
from enum import Enum
import pandas as pd
from .consts import *
from zoneinfo import ZoneInfo

TZ = {
    "Lidar(Tomsk)": ZoneInfo("Asia/Tomsk"),
}


def scale(x):
    return np.log(1 + 0.0005 * x)


def descale(x):
    return (np.exp(x) - 1) / 0.0005


def graph_to_row(altitudes, x, y):
    values = np.interp(altitudes, x, y)
    return values


def angle_difference(angle1: float, angle2: float) -> float:
    dangle = angle2 - angle1
    if dangle > 180:
        dangle -= 360
    if dangle < -180:
        dangle += 360
    return dangle


MAX_ALTITUDE = scale(14000)
MIN_ALTITUDE = scale(0)
NUM_ALTITUDE = 31
ALTITUDES = descale(np.linspace(MIN_ALTITUDE, MAX_ALTITUDE, NUM_ALTITUDE))

Place = namedtuple("Place", ["id", "latitude", "longitude", "name"])
Era5Data = namedtuple("Date", ["id", "place_id", "timestamp", "date"])
LidarData = namedtuple(
    "LidarData",
    [
        "id",
        "place_id",
        "timestamp_started_at",
        "timestamp_finished_at",
        "start",
        "end",
        "scattering",
        "thickness",
        "parallel",
        "sort",
        "category",
        "date_started_at",
        "date_finished_at",
    ],
)
AeroData = namedtuple("AeroData", ["id", "place_id", "elevation", "timestamp", "date"])
SunData = namedtuple(
    "SunData", ["id", "place_id", "timestamp", "azimuth", "zenith", "date"]
)

Merra2Collection = namedtuple("Merra2Collection", ["id", "place_id", "path"])
Merra2Data = namedtuple("Merra2Data", ["id", "collection_id", "timestamp", "date"])


class DB:
    def __init__(self, dbpath: str):
        self._dbpath = dbpath
        self._connection = None

    def __enter__(self):
        self._connection = sqlite3.connect(self._dbpath)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._connection:
            self._connection.close()
            self._connection = None
            print("closed")

    def get_places(self):
        places = []
        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            for row in con.execute(
                "SELECT id, latitude, longitude, name FROM places ORDER BY id"
            ):
                p = Place(*row)
                places.append(p)
        finally:
            if self._connection is None:
                con.close()
        return places

    def get_place(self, name: str):
        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            for row in con.execute(
                "SELECT id, latitude, longitude, name FROM places WHERE name = ? ORDER BY id",
                (name,),
            ):
                p = Place(*row)
                return p
        finally:
            if self._connection is None:
                con.close()
        raise Exception("There is no %s" % name)

    def get_era5_data(self, place: Place):
        data = []
        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            for row in con.execute(
                "SELECT id, place_id, date FROM era5_data WHERE place_id = ? ORDER BY date",
                (place.id,),
            ):
                date = datetime.datetime.fromtimestamp(
                    row[-1], tz=datetime.timezone.utc
                )
                data.append(Era5Data(*row, date))
        finally:
            if self._connection is None:
                con.close()
        return data

    def get_era5_measurements(
        self, data: Era5Data, measure: Measure, *, altitude=17000
    ):
        """
        Получить данные измерения

        altitude - если скаляр, то максимальная высота, если список/кортеж, то сетка высот (интерполируется)
        """
        field = None
        if measure == Measure.PRESSURE:
            field = "pressure"
        elif measure == Measure.TEMPERATURE:
            field = "temperature"
        elif measure == Measure.REL_HUMIDITY:
            field = "rel_humidity"
        elif measure == Measure.ABS_HUMIDITY:
            field = "abs_humidity"
        elif measure == Measure.WIND_DIRECTION:
            field = "wind_direction"
        elif measure == Measure.WIND_SPEED:
            field = "wind_speed"
        else:
            raise Exception(measure)

        alts = []
        vals = []

        is_scalar = isinstance(altitude, float) or isinstance(altitude, int)
        try:
            con = self._connection or sqlite3.connect(self._dbpath)

            if is_scalar:
                expr = con.execute(
                    "SELECT id, altitude, %s FROM era5_measurements WHERE data_id = ? and altitude <= ? ORDER BY altitude"
                    % field,
                    (data.id, altitude),
                )
            else:
                expr = con.execute(
                    "SELECT id, altitude, %s FROM era5_measurements WHERE data_id = ? ORDER BY altitude"
                    % field,
                    (data.id,),
                )
            for row in expr:
                alts.append(row[1])
                vals.append(row[2])
            if not is_scalar and altitude is not None:
                new_alts = altitude
                new_vals = graph_to_row(np.asarray(new_alts), alts, vals)
                alts = new_alts
                vals = new_vals
        finally:
            if self._connection is None:
                con.close()
        return alts, vals

    def get_era5_measurements_approx(
        self, place: Place, measure: Measure, timestamp: int, *, altitudes: list[float]
    ):
        """
        Получить данные реанализа приблизительно через линейную интерполяцию соседних измерений
        в любой момент времени (timestamp) и для заданного набора высот (altitudes)

        altitude - любой массив высот
        """
        timestamp = int(
            timestamp
        )  # Для того, чтобы избежать странных проблем из-за numpy
        first = None
        second = None

        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            # con.set_trace_callback(print)
            for row in con.execute(
                "SELECT id, place_id, date FROM era5_data WHERE place_id = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                (place.id, timestamp),
            ):
                date = datetime.datetime.fromtimestamp(
                    row[-1], tz=datetime.timezone.utc
                )
                first = Era5Data(*row, date)

            for row in con.execute(
                "SELECT id, place_id, date FROM era5_data WHERE place_id = ? AND date >= ? ORDER BY date LIMIT 1",
                (place.id, timestamp),
            ):
                date = datetime.datetime.fromtimestamp(
                    row[-1], tz=datetime.timezone.utc
                )
                second = Era5Data(*row, date)
        finally:
            if self._connection is None:
                con.close()

        if first is None or second is None:
            return [], []

        # Левый край
        fX, fY = self.get_era5_measurements(first, measure, altitude=altitudes)
        if first.timestamp == second.timestamp:
            return fX, fY

        # Правый край
        sX, sY = self.get_era5_measurements(second, measure, altitude=altitudes)

        # Линейная интерполяция
        dt = second.timestamp - first.timestamp
        if dt > 60 * 60 * 1.5:
            print(f"WARN: For timestamp [{timestamp}] got time delta in ERA5 = {dt}")
        t = timestamp - first.timestamp
        mY = []
        for y1, y2 in zip(fY, sY):
            dy = y2 - y1
            yc = y1 + dy * (t / dt)
            mY.append(yc)
        return fX, mY

    def get_lidar_data(
        self, place: Place, *, category: Category = Category.HIGH_LEVEL_CLOUD
    ):
        """
        Получить список описаний лидарных данных. По умолчанию, только для хороших данных.
        Если нужны все данные , то можно передать category=None.
        Если нужные данные только с высоким шумом, то category=Category.LOW_SIGNAL
        """
        tz = TZ[place.name]
        data = []
        try:
            con = self._connection or sqlite3.connect(self._dbpath)

            stmt = """
                SELECT
                    id, place_id, started_at, finished_at, start, end, scattering, thickness, parallel, sort, category_id
                FROM lidar_data WHERE place_id = ?"""
            flt = [
                place.id,
            ]
            if category is not None:
                stmt += " AND category_id = ?"
                flt.append(category.value)
            stmt += " ORDER BY started_at"

            for row in con.execute(stmt, flt):
                # date_started_at = datetime.datetime.fromtimestamp(row[2], tz=datetime.timezone.utc)
                # date_finished_at = datetime.datetime.fromtimestamp(row[3], tz=datetime.timezone.utc)
                date_started_at = datetime.datetime.fromtimestamp(row[2], tz=tz)
                date_finished_at = datetime.datetime.fromtimestamp(row[3], tz=tz)
                values = list(row)
                if values[-1] == 1:
                    values[-1] = Category.HIGH_LEVEL_CLOUD
                elif values[-1] == 2:
                    values[-1] = Category.LOW_SIGNAL
                elif values[-1] == 0:
                    values[-1] = Category.CLOUDLESS
                else:
                    raise Exception("Unknown Category")
                data.append(LidarData(*values, date_started_at, date_finished_at))
        finally:
            if self._connection is None:
                con.close()
        return data

    def get_lidar_measurements(
        self, data: LidarData, channel: Channel, *, only_in_border=True
    ) -> tuple[list[float], list[float], list[list[float]]]:
        """
        Получить измерения лидара для указанной даты и канала.

        only_in_border - если True, то загрузить только в пределах толщи ОВЯ
        """
        xs = []
        snrs = []
        ms = []

        try:
            con = self._connection or sqlite3.connect(self._dbpath)

            if only_in_border:
                query = con.execute(
                    """
                    SELECT
                        altitude,
                        snr,
                        m01, m02, m03, m04,
                        m05, m06, m07, m08,
                        m09, m10, m11, m12,
                        m13, m14, m15, m16
                    FROM lidar_measurements
                    WHERE data_id = ? AND channel_id = ? AND altitude >= ? AND altitude <= ?
                    ORDER BY altitude
                """,
                    (data.id, channel.value, data.start, data.end),
                )
            else:
                query = con.execute(
                    """
                    SELECT
                        altitude,
                        snr,
                        m01, m02, m03, m04,
                        m05, m06, m07, m08,
                        m09, m10, m11, m12,
                        m13, m14, m15, m16
                    FROM lidar_measurements
                    WHERE data_id = ? AND channel_id = ?
                    ORDER BY altitude
                """,
                    (data.id, channel.value),
                )
            for row in query:
                h, snr, *m = row
                xs.append(h)
                snrs.append(snr)
                ms.append(m)
        finally:
            if self._connection is None:
                con.close()

        return xs, snrs, ms

    def get_lidar_dataframe(
        self,
        place: Place,
        *,
        only_in_border: bool = True,
        category: Category = Category.HIGH_LEVEL_CLOUD,
    ):
        table = {
            k: []
            for k in (
                "date_started_at",
                "date_finished_at",
                "timestamp_started_at",
                "timestamp_finished_at",
                "altitude",
                "channel",
                "snr",
                *["m%02d" % (i + 1) for i in range(16)],
            )
        }
        data = self.get_lidar_data(place, category=category)

        for d in data:
            for channel in Channel:
                x, snr, m = self.get_lidar_measurements(
                    d, channel, only_in_border=only_in_border
                )
                for xv, sv, mv in zip(x, snr, m):
                    table["date_started_at"].append(d.date_started_at)
                    table["date_finished_at"].append(d.date_finished_at)
                    table["timestamp_started_at"].append(d.timestamp_started_at)
                    table["timestamp_finished_at"].append(d.timestamp_finished_at)
                    table["altitude"].append(xv)
                    table["channel"].append(channel)
                    table["snr"].append(sv)
                    for i in range(16):
                        table["m%02d" % (i + 1)].append(mv[i])

        return pd.DataFrame(table)

    def get_aero_data(self, place: Place) -> list[AeroData]:
        data = []
        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            for row in con.execute(
                "SELECT id, place_id, elevation, date FROM aero_data WHERE place_id = ? ORDER BY date",
                (place.id,),
            ):
                date = datetime.datetime.fromtimestamp(
                    row[-1], tz=datetime.timezone.utc
                )
                data.append(AeroData(*row, date))
        finally:
            if self._connection is None:
                con.close()
        return data

    def get_aero_measurements(self, data: AeroData, measure: Measure, *, altitude=None):
        """
        Получить данные измерения

        altitude - максимальная допустимая высота (None если нужны все), либо сетка высот
        """
        field = None
        if measure == Measure.PRESSURE:
            field = "pressure"
        elif measure == Measure.TEMPERATURE:
            field = "temperature"
        elif measure == Measure.REL_HUMIDITY:
            field = "rel_humidity"
        elif measure == Measure.ABS_HUMIDITY:
            field = "abs_humidity"
        elif measure == Measure.WIND_DIRECTION:
            field = "wind_direction"
        elif measure == Measure.WIND_SPEED:
            field = "wind_speed"
        else:
            raise Exception(measure)

        alts = []
        vals = []

        is_scalar = isinstance(altitude, float) or isinstance(altitude, int)
        try:
            con = self._connection or sqlite3.connect(self._dbpath)

            if is_scalar:
                expr = con.execute(
                    "SELECT id, altitude, %s FROM aero_measurements WHERE data_id = ? and altitude <= ? ORDER BY altitude"
                    % field,
                    (data.id, altitude),
                )
            else:
                expr = con.execute(
                    "SELECT id, altitude, %s FROM aero_measurements WHERE data_id = ? ORDER BY altitude"
                    % field,
                    (data.id,),
                )

            for row in expr:
                alts.append(row[1])
                vals.append(row[2])
            if not is_scalar and altitude is not None:
                new_alts = altitude
                new_vals = graph_to_row(np.asarray(new_alts), alts, vals)
                alts = new_alts
                vals = new_vals
        finally:
            if self._connection is None:
                con.close()
        return alts, vals

    def get_sun_measurements(self, place: Place) -> list[SunData]:
        """
        Загружаем все данные по положению солнца
        """
        result = []
        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            rows = con.execute(
                "SELECT id, place_id, date, azimuth, zenith FROM sun_measurements WHERE place_id = ? ORDER BY date",
                (place.id,),
            )

            for row in rows:
                date = datetime.datetime.fromtimestamp(row[2], tz=datetime.timezone.utc)
                result.append(SunData(*row, date))
        finally:
            if self._connection is None:
                con.close()
        return result

    def get_sun_measurements_approx(self, place: Place, timestamp: int) -> SunData:
        """
        Загружаем в конкретное время (линейная интерполяция)
        """

        first = None
        second = None
        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            for row in con.execute(
                "SELECT id, place_id, date, azimuth, zenith FROM sun_measurements WHERE place_id = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                (place.id, timestamp),
            ):
                date = datetime.datetime.fromtimestamp(row[2], tz=datetime.timezone.utc)
                first = SunData(*row, date)

            for row in con.execute(
                "SELECT id, place_id, date, azimuth, zenith FROM sun_measurements WHERE place_id = ? AND date >= ? ORDER BY date LIMIT 1",
                (place.id, timestamp),
            ):
                date = datetime.datetime.fromtimestamp(row[2], tz=datetime.timezone.utc)
                second = SunData(*row, date)
        finally:
            if self._connection is None:
                con.close()

        if first is None or second is None:
            raise Exception("Invalid time stamp or place. Cannot get any data.")

        if first.timestamp == timestamp:
            return first

        dt = second.timestamp - first.timestamp
        t = timestamp - first.timestamp
        date = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        azimuth = first.azimuth + angle_difference(first.azimuth, second.azimuth) * (
            t / dt
        )
        zenith = first.zenith + angle_difference(first.zenith, second.zenith) * (t / dt)

        return SunData(
            [first.id, second.id], place.id, timestamp, azimuth, zenith, date
        )

    def get_merra2_collections(self, place: Place) -> list[Merra2Collection]:
        """
        Загружаем все данные по коллекциям данных из merra2
        """

        result = []
        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            rows = con.execute(
                "SELECT id, place_id, path FROM merra2_collections WHERE place_id = ? ORDER BY path",
                (place.id,),
            )

            for row in rows:
                result.append(Merra2Collection(*row))
        finally:
            if self._connection is None:
                con.close()
        return result

    def get_merra2_collection(self, place: Place, path: str) -> Merra2Collection:
        """
        Загружаем конкретную коллекцию
        """
        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            for row in con.execute(
                "SELECT id, place_id, path FROM merra2_collections WHERE place_id = ? AND path = ?",
                (place.id, path),
            ):
                p = Merra2Collection(*row)
                return p
        finally:
            if self._connection is None:
                con.close()
        raise Exception("There is no %s" % path)

    def get_merra2_data(self, collection: Merra2Collection) -> list[Merra2Data]:
        """
        Получить описание всех данных по указанной коллекции
        """

        result = []
        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            rows = con.execute(
                "SELECT id, collection_id, date FROM merra2_data WHERE collection_id = ? ORDER BY date",
                (collection.id,),
            )

            for row in rows:
                date = datetime.datetime.fromtimestamp(row[2], tz=datetime.timezone.utc)
                result.append(Merra2Data(*row, date))
        finally:
            if self._connection is None:
                con.close()
        return result

    def get_merra2_measurements(
        self, data: Merra2Data
    ) -> tuple[list[float], list[float]]:
        """
        Возвращает два списка: уровни давлений и значения величины на этих уровнях
        """

        p = []
        v = []

        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            rows = con.execute(
                "SELECT pressure, value FROM merra2_measurements WHERE data_id = ? ORDER BY pressure DESC",
                (data.id,),
            )

            for row in rows:
                p.append(row[0])
                v.append(row[1])
        finally:
            if self._connection is None:
                con.close()
        return p, v

    def get_merra2_measurements_approx(
        self, collection: Merra2Collection, timestamp: int
    ) -> tuple[list[float], list[float]]:
        timestamp = int(
            timestamp
        )  # Для того, чтобы избежать странных проблем из-за numpy
        first = None
        second = None

        try:
            con = self._connection or sqlite3.connect(self._dbpath)
            # con.set_trace_callback(print)
            for row in con.execute(
                "SELECT id, collection_id, date FROM merra2_data WHERE collection_id = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                (collection.id, timestamp),
            ):
                date = datetime.datetime.fromtimestamp(
                    row[-1], tz=datetime.timezone.utc
                )
                first = Merra2Data(*row, date)

            for row in con.execute(
                "SELECT id, collection_id, date FROM merra2_data WHERE collection_id = ? AND date >= ? ORDER BY date LIMIT 1",
                (collection.id, timestamp),
            ):
                date = datetime.datetime.fromtimestamp(
                    row[-1], tz=datetime.timezone.utc
                )
                second = Merra2Data(*row, date)
        finally:
            if self._connection is None:
                con.close()

        if first is None:
            raise Exception("Cannot find left border of interval")
        if second is None:
            raise Exception("Cannot find right border of interval")

        dt = second.timestamp - first.timestamp
        if dt > 60 * 60 * 3.5:
            raise Exception("Too large interval to interpolate")

        # Левый край
        fP, fV = self.get_merra2_measurements(first)
        if first.timestamp == second.timestamp:
            return fP, fV

        # Правый край
        sP, sV = self.get_merra2_measurements(second)

        assert len(fP) == len(sP)
        for ff, ss in zip(fP, sP):
            assert ff == ss

        # Линейная интерполяция
        t = timestamp - first.timestamp
        mV = []
        for y1, y2 in zip(fV, sV):
            dy = y2 - y1
            yc = y1 + dy * (t / dt)
            mV.append(yc)
        return fP, mV
