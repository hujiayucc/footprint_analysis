const api_url = () => {
  return document.querySelector('meta[name="api_url"]').content;
}

async function analyzeFootprint() {
    const formData = {
        foot_length: parseFloat(document.getElementById('foot_length').value),
        foot_width: parseFloat(document.getElementById('foot_width').value),
        arch_height: parseFloat(document.getElementById('arch_height').value),
        depth_diff: parseFloat(document.getElementById('depth_diff').value),
        pressure_offset_x: parseFloat(document.getElementById('pressure_offset_x').value),
        ground_type: document.getElementById('ground_type').value,
        humidity: parseInt(document.getElementById('humidity').value),
        depth: parseFloat(document.getElementById('depth').value),
        pressure_avg: parseFloat(document.getElementById('pressure_avg').value)
    };

    try {
        const response = await fetch(api_url(), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) throw new Error(`HTTP错误: ${response.status}`);

        const result = await response.json();
        updateResults(result);
    } catch (error) {
        console.error('分析失败:', error);
        alert(`分析失败: ${error.message}`);
    }
}

function updateResults(data) {
    // 更新基础信息
    document.getElementById('heightResult').textContent = data.height.toFixed(1);
    document.getElementById('weightResult').textContent = data.weight.toFixed(1);

    // 更新腿型概率
    const probabilities = data.leg_type_probs;
    document.querySelectorAll('.leg-type').forEach((item, index) => {
        const percent = (probabilities[index] * 100).toFixed(1) + '%';
        item.querySelector('.prob-value').textContent = percent;
        item.querySelector('.progress-fill').style.width = percent;
    });

    // 显示结果区域
    document.getElementById('resultSection').style.display = 'block';
}

const sampleData = [
    {
        foot_length: 27.4,
        foot_width: 10.8,
        arch_height: 2.7,
        depth_diff: 2.1,
        pressure_offset_x: -1.6,
        ground_type: "wet_soil",
        humidity: 54,
        depth: 4.1,
        pressure_avg: 100.0
    },
    {
        foot_length: 27.3,
        foot_width: 11.6,
        arch_height: 2.5,
        depth_diff: 2.8,
        pressure_offset_x: -2.3,
        ground_type: "concrete",
        humidity: 50,
        depth: 0.4,
        pressure_avg: 100.0
    },
    {
        foot_length: 27.3,
        foot_width: 9.5,
        arch_height: 2.9,
        depth_diff: 2.9,
        pressure_offset_x: -1.3,
        ground_type: "grass",
        humidity: 26,
        depth: 1.6,
        pressure_avg: 91.1
    },
    {
        foot_length: 27.4,
        foot_width: 10.9,
        arch_height: 2.6,
        depth_diff: 2.7,
        pressure_offset_x: -1.5,
        ground_type: "dry_soil",
        humidity: 52,
        depth: 2.5,
        pressure_avg: 100.0
    },
    {
        foot_length: 27.4,
        foot_width: 9.2,
        arch_height: 2.8,
        depth_diff: 1.8,
        pressure_offset_x: -2.3,
        ground_type: "sand",
        humidity: 47,
        depth: 3.4,
        pressure_avg: 88.2
    }
];

let currentDataIndex = 0;

function fillSampleData() {
    const data = sampleData[currentDataIndex];

    // 填充数值型输入
    document.getElementById('foot_length').value = data.foot_length;
    document.getElementById('foot_width').value = data.foot_width;
    document.getElementById('arch_height').value = data.arch_height;
    document.getElementById('depth_diff').value = data.depth_diff;
    document.getElementById('pressure_offset_x').value = data.pressure_offset_x;
    document.getElementById('humidity').value = data.humidity;
    document.getElementById('depth').value = data.depth;
    document.getElementById('pressure_avg').value = data.pressure_avg;

    // 设置下拉菜单
    const groundTypeSelect = document.getElementById('ground_type');
    groundTypeSelect.value = data.ground_type;

    // 更新索引（循环使用）
    currentDataIndex = (currentDataIndex + 1) % sampleData.length;
}

function resetForm() {
    // 清空所有输入
    document.querySelectorAll('input').forEach(input => {
        if (input.type === 'number') input.value = '';
    });

    // 重置下拉菜单
    document.getElementById('ground_type').selectedIndex = 0;

    // 隐藏结果
    document.getElementById('resultSection').style.display = 'none';

    // 重置索引
    currentDataIndex = 0;
}