// NEX-GDDP-CMIP6: daily tasmax (+ hurs) for 2031-2040, per upazila, per GCM,
// for SSP2-4.5 and SSP5-8.5. Paste into the Earth Engine Code Editor.
//
// 1. Upload data/boundaries/study_area_upazilas.gpkg (or the .shp) as an
//    asset and put its path below. The GID_3 property is the join key.
// 2. Run; one export task per scenario appears under Tasks. Each CSV has
//    columns GID_3, date, model, scenario, tasmax, hurs.
// 3. Drop the CSVs anywhere under data/ (not boundaries/ or cache/). The
//    dashboard lists them under "Projection exports".
//
// Do not average models before running this: the dashboard computes the six
// indices per model and summarises the ensemble afterwards.

var zones = ee.FeatureCollection('users/YOUR_ASSET/study_area_upazilas');

var START = '2031-01-01', END = '2041-01-01';   // 2031-2040 inclusive
var SEASON = [3, 6];                             // warm season, months
var MODELS = ['ACCESS-CM2', 'EC-Earth3', 'GFDL-ESM4', 'INM-CM5-0',
              'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0',
              'NorESM2-MM', 'UKESM1-0-LL'];
var SCENARIOS = ['ssp245', 'ssp585'];

// 0.25 deg cells (~27 km) are larger than most upazilas; a zonal mean at the
// native scale is the honest reduction. Expect spatially smooth hazard.
var SCALE = 27830;

function exportScenario(scenario) {
  var perModel = MODELS.map(function (model) {
    var coll = ee.ImageCollection('NASA/GDDP-CMIP6')
      .filter(ee.Filter.eq('model', model))
      .filter(ee.Filter.eq('scenario', scenario))
      .filterDate(START, END)
      .filter(ee.Filter.calendarRange(SEASON[0], SEASON[1], 'month'))
      .select(['tasmax', 'hurs']);

    return coll.map(function (img) {
      var stats = img.reduceRegions({
        collection: zones,
        reducer: ee.Reducer.mean(),
        scale: SCALE,
        tileScale: 4
      });
      var date = img.date().format('YYYY-MM-dd');
      return stats.map(function (f) {
        return f.set({date: date, model: model, scenario: scenario});
      });
    }).flatten();
  });

  var rows = ee.FeatureCollection(perModel).flatten();

  Export.table.toDrive({
    collection: rows,
    description: 'cmip6_' + scenario + '_2031_2040_khulna',
    fileNamePrefix: 'cmip6_' + scenario + '_2031_2040_khulna',
    fileFormat: 'CSV',
    selectors: ['GID_3', 'date', 'model', 'scenario', 'tasmax', 'hurs']
  });
}

SCENARIOS.forEach(exportScenario);

// Optional sanity check: one model, one day, on the map.
var sample = ee.ImageCollection('NASA/GDDP-CMIP6')
  .filter(ee.Filter.eq('model', 'EC-Earth3'))
  .filter(ee.Filter.eq('scenario', 'ssp585'))
  .filterDate('2035-04-15', '2035-04-16').first().select('tasmax');
Map.centerObject(zones, 8);
Map.addLayer(sample.subtract(273.15).clip(zones.geometry()),
             {min: 30, max: 42, palette: ['ffe5dd', 'e87652', '7a2302']}, 'tasmax degC');
Map.addLayer(zones.style({color: '14181D', fillColor: '00000000', width: 1}), {}, 'upazilas');
