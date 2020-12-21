const fs = require('fs');
const path = require('path');

const defaultDatapointPolicies = {
    // Median is a safe default to use, see https://doc.calcite.siliceum.com/performance/stability/statistics.html#descriptive-statistics for why we do not pick the mean.
    aggregationPolicy: 'median',
    // We want to consider only significant changes, in our case meaning 20% increase of the median.
    // We chose 20% as a safe default since even when stabilized, a CPU can have such variations due to (for example) the Intel JCC Erratum. (https://www.intel.com/content/dam/support/us/en/documents/processors/mitigations-jump-conditional-code-erratum.pdf)
    // We can at a later point in time lower this value if benchmarks are stable enough.
    // See https://doc.calcite.siliceum.com/performance/stability/index.html for a reference of what can impact stability.
    diffPolicy: 'relativeDifference',
    regressionPolicy: 'lessIsBetter',
    regressionArgument: 20
};

const fromGoogleBenchFile = (filePath, builder, calciteContext) => {
    const data = calciteContext.readJson(filePath);
    
    const testSuiteName = path.basename(data.context.executable);

    const benchsNoAggregate = data.benchmarks.filter(bench => bench.run_type === 'iteration');
    benchsNoAggregate.forEach((bench) => {
        builder.addOrUpdateDataPoint(
            testSuiteName ,
            bench.run_name,
            'real_time', {
                values: [bench['real_time']],
                unit: bench['time_unit'],
                ...defaultDatapointPolicies
            }
        );

        builder.addOrUpdateDataPoint(
            testSuiteName ,
            bench.run_name,
            'cpu_time', {
                values: [bench['cpu_time']],
                unit: bench['time_unit'],
                ...defaultDatapointPolicies
            }
        );
    });
};

// TODO: use https://www.npmjs.com/package/glob ?
async function buildFilesList(pathsList, fileRegex)
{
    const pathsArray = pathsList.split(path.delimiter);
    return (await Promise.all( pathsArray.map( async pathEntry => {
        try{
            const filesInPath = await fs.promises.readdir(pathEntry, { withFileTypes: true });
            const test = filesInPath.map( file => 
            {
                if(file.isDirectory())
                {
                    // no recursion allowed
                }
                else if(file.isFile() && fileRegex.test(file.name))
                {
                    return path.join(pathEntry, file.name);
                }
                return null;
            }
            ).filter( x => x !== null );
            return test;

        } catch(_)
        {
            return [pathEntry];
        }
    }))).flat();
}

module.exports = async function (context) {
    if (!process.env.BENCH_OUTPUT_FILES) {
        console.error('[Error] - The BENCH_OUTPUT_FILES env variable was not set');
        process.exit(1);
    }

    const benchFiles = await buildFilesList(process.env.BENCH_OUTPUT_FILES, /.*\.json$/);
    
    const builder = context.helpers.TestSuitesBuilderFactory();
    benchFiles.forEach(filePath => fromGoogleBenchFile(filePath, builder, context.helpers));

    return {
        testSuites: builder.assembleAndGetAll()
    };
};
