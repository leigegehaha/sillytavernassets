/**
 * 向量检索上下文管理系统
 * 用于减少token消耗，增强AI记忆力
 */

class ContextVectorManager {
    constructor() {
        this.conversationEmbeddings = []; // 存储每轮对话的向量和元数据
        this.embeddingMethod = 'keyword'; // 'keyword' | 'api' | 'transformers'
        this.maxRetrieveCount = 5; // 最多检索5条相关历史
        this.minSimilarityThreshold = 0.3; // 最低相似度阈值
    }

    /**
     * 【方案1】关键词权重法（默认，无需API）
     * 使用TF-IDF提取关键词，计算余弦相似度
     */
    extractKeywords(text) {
        // 1. 分词（简单按字符分割，可优化为jieba分词）
        const words = text.match(/[\u4e00-\u9fa5]+|[a-zA-Z]+/g) || [];
        
        // 2. 计算词频（TF）
        const wordFreq = {};
        words.forEach(word => {
            if (word.length > 1) { // 过滤单字
                wordFreq[word] = (wordFreq[word] || 0) + 1;
            }
        });
        
        // 3. 提取高频词作为关键词
        const keywords = Object.entries(wordFreq)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 20) // 取前20个关键词
            .map(([word, freq]) => ({ word, weight: freq }));
        
        return keywords;
    }

    /**
     * 创建简单向量（关键词权重向量）
     */
    createKeywordVector(text) {
        const keywords = this.extractKeywords(text);
        const vector = {};
        
        // 构建稀疏向量
        keywords.forEach(({ word, weight }) => {
            vector[word] = weight;
        });
        
        return vector;
    }

    /**
     * 计算余弦相似度（支持稀疏向量对象和稠密向量数组）
     */
    calculateCosineSimilarity(vec1, vec2) {
        // 空值检查
        if (!vec1 || !vec2) {
            console.warn('[相似度计算] 向量为空');
            return 0;
        }
        
        // 判断向量类型
        const isArray1 = Array.isArray(vec1);
        const isArray2 = Array.isArray(vec2);
        
        // 如果类型不匹配，尝试转换
        if (isArray1 !== isArray2) {
            console.warn('[相似度计算] 向量类型不匹配，尝试转换');
            // 如果一个是数组一个是对象，无法比较，返回0
            return 0;
        }
        
        if (isArray1 && isArray2) {
            // 稠密向量（数组）相似度计算
            return this.calculateArrayCosineSimilarity(vec1, vec2);
        } else {
            // 稀疏向量（对象）相似度计算
            return this.calculateObjectCosineSimilarity(vec1, vec2);
        }
    }
    
    /**
     * 计算对象形式的稀疏向量相似度
     */
    calculateObjectCosineSimilarity(vec1, vec2) {
        const allKeys = new Set([...Object.keys(vec1), ...Object.keys(vec2)]);
        
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        
        allKeys.forEach(key => {
            const v1 = vec1[key] || 0;
            const v2 = vec2[key] || 0;
            dotProduct += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        });
        
        if (norm1 === 0 || norm2 === 0) return 0;
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    
    /**
     * 计算数组形式的稠密向量相似度
     */
    calculateArrayCosineSimilarity(vec1, vec2) {
        const len = Math.min(vec1.length, vec2.length);
        
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        
        for (let i = 0; i < len; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        
        if (norm1 === 0 || norm2 === 0) return 0;
        
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    /**
     * 添加对话到向量库
     */
    async addConversation(userMessage, aiResponse, turnIndex, variables) {
        // 🔍 检查是否已存在相同的 turnIndex
        const existingIndex = this.conversationEmbeddings.findIndex(
            conv => conv.turnIndex === turnIndex
        );
        
        if (existingIndex !== -1) {
            console.warn(`[向量库] ⚠️ turnIndex ${turnIndex} 已存在，将覆盖旧数据`);
            // 删除旧的记录
            this.conversationEmbeddings.splice(existingIndex, 1);
        }
        
        let vector;
        
        // 合并用户消息和AI回复作为一个语义单元
        const combinedText = `${userMessage}\n${aiResponse}`;
        
        try {
            if (this.embeddingMethod === 'keyword') {
                // 方案1：关键词向量
                vector = this.createKeywordVector(combinedText);
            } else if (this.embeddingMethod === 'api') {
                // 方案2：调用API获取embedding
                vector = await this.getEmbeddingFromAPI(combinedText);
            } else if (this.embeddingMethod === 'transformers') {
                // 方案3：浏览器端模型（需要加载transformers.js）
                vector = await this.getEmbeddingFromTransformers(combinedText);
            } else {
                // 默认使用关键词方法
                console.warn(`[向量库] 未知的向量化方法：${this.embeddingMethod}，使用关键词方法`);
                vector = this.createKeywordVector(combinedText);
            }
            
            // 验证向量
            if (!vector || (Array.isArray(vector) && vector.length === 0) || (typeof vector === 'object' && Object.keys(vector).length === 0)) {
                console.error('[向量库] 向量生成失败，使用关键词方法作为后备');
                vector = this.createKeywordVector(combinedText);
            }
        } catch (error) {
            console.error('[向量库] 向量化失败:', error);
            // 回退到关键词方法
            vector = this.createKeywordVector(combinedText);
        }
        
        // 提取关键信息摘要
        const summary = this.extractSummary(userMessage, aiResponse, variables);
        
        this.conversationEmbeddings.push({
            turnIndex: turnIndex,
            userMessage: userMessage,
            aiResponse: aiResponse,
            vector: vector,
            vectorType: Array.isArray(vector) ? 'dense' : 'sparse', // 标记向量类型
            summary: summary,
            timestamp: Date.now(),
            variables: this.extractImportantVariables(variables)
        });
        
        console.log(`[向量库] 已添加第${turnIndex}轮对话（方法：${this.embeddingMethod}），当前库大小：${this.conversationEmbeddings.length}`);
    }

    /**
     * 提取对话摘要（关键信息）
     */
    extractSummary(userMessage, aiResponse, variables) {
        const summary = [];
        
        // 提取用户行动
        if (userMessage.length < 50) {
            summary.push(`玩家：${userMessage}`);
        } else {
            summary.push(`玩家：${userMessage.substring(0, 50)}...`);
        }
        
        // 提取AI回复关键词
        const keywords = this.extractKeywords(aiResponse);
        if (keywords.length > 0) {
            const topKeywords = keywords.slice(0, 5).map(k => k.word).join('、');
            summary.push(`关键词：${topKeywords}`);
        }
        
        // 提取重要变量变化
        if (variables.location) {
            summary.push(`地点：${variables.location}`);
        }
        
        return summary.join(' | ');
    }

    /**
     * 提取重要变量（用于快速回忆）
     */
    extractImportantVariables(variables) {
        return {
            location: variables.location,
            realm: variables.realm,
            hp: variables.hp,
            mp: variables.mp,
            // 只保存关键信息，减少存储
            hasNewItems: variables.items && variables.items.length > 0,
            hasNewRelationships: variables.relationships && variables.relationships.length > 0
        };
    }

    /**
     * 🆕 智能检测向量库的主要向量类型
     */
    detectVectorType() {
        if (this.conversationEmbeddings.length === 0) return 'keyword';
        
        // 统计向量类型
        const typeCounts = {
            dense: 0,  // 稠密向量（数组）
            sparse: 0  // 稀疏向量（对象）
        };
        
        this.conversationEmbeddings.forEach(conv => {
            if (Array.isArray(conv.vector)) {
                typeCounts.dense++;
            } else if (typeof conv.vector === 'object') {
                typeCounts.sparse++;
            }
        });
        
        // 返回占比最大的类型
        return typeCounts.dense > typeCounts.sparse ? 'dense' : 'sparse';
    }

    /**
     * 检索相关上下文（智能兼容版）
     */
    async retrieveRelevantContext(currentInput, recentHistory = []) {
        if (this.conversationEmbeddings.length === 0) {
            return {
                relevantChunks: [],
                recentChunks: recentHistory
            };
        }
        
        try {
            // 🆕 1. 智能检测向量库类型
            const vectorLibType = this.detectVectorType();
            console.log(`[向量检索] 向量库类型：${vectorLibType}，当前方法：${this.embeddingMethod}`);
            
            // 🆕 2. 根据向量库类型和当前设置，智能选择检索策略
            let currentVector;
            let useArrayVector = false;  // 标记是否使用数组向量
            
            if (vectorLibType === 'dense' && this.embeddingMethod !== 'keyword') {
                // 向量库是稠密向量，且当前不是关键词模式
                // 策略：尝试生成稠密向量，失败则降级到关键词
                console.log('[向量检索] 尝试生成稠密向量...');
                
                try {
                    if (this.embeddingMethod === 'api') {
                        // 调用API生成向量
                        currentVector = await this.getEmbeddingFromAPI(currentInput);
                        useArrayVector = true;
                        console.log('[向量检索] ✅ 使用API生成稠密向量');
                    } else if (this.embeddingMethod === 'transformers') {
                        // 调用浏览器端模型生成向量
                        currentVector = await this.getEmbeddingFromTransformers(currentInput);
                        useArrayVector = true;
                        console.log('[向量检索] ✅ 使用Transformers生成稠密向量');
                    } else {
                        // 未知方法，降级
                        throw new Error('未知的向量化方法');
                    }
                    
                    // 验证向量是否有效
                    if (!currentVector || !Array.isArray(currentVector) || currentVector.length === 0) {
                        throw new Error('生成的向量无效');
                    }
                } catch (error) {
                    // 生成失败，降级到关键词方法
                    console.warn('[向量检索] ⚠️ 稠密向量生成失败，降级使用关键词方法:', error.message);
                    currentVector = this.createKeywordVector(currentInput);
                    useArrayVector = false;
                }
            } else {
                // 默认使用关键词方法（最兼容）
                currentVector = this.createKeywordVector(currentInput);
                useArrayVector = false;
            }
            
            // 验证向量
            if (!currentVector || (Array.isArray(currentVector) ? currentVector.length === 0 : Object.keys(currentVector).length === 0)) {
                console.warn('[向量检索] 当前输入向量为空，跳过检索');
                return {
                    relevantChunks: [],
                    recentChunks: recentHistory
                };
            }
            
            // 🆕 3. 智能计算相似度（支持混合向量类型）
            const similarities = this.conversationEmbeddings.map((conv, index) => {
                let convVector = conv.vector;
                let similarity = 0;
                
                const isConvArray = Array.isArray(conv.vector);
                const isCurrArray = Array.isArray(currentVector);
                
                if (isConvArray === isCurrArray) {
                    // 类型匹配，直接计算
                    similarity = this.calculateCosineSimilarity(currentVector, convVector);
                } else {
                    // 🆕 类型不匹配，智能转换
                    if (isConvArray && !isCurrArray) {
                        // 库中是数组，当前是对象 -> 将库中向量转为关键词向量
                        convVector = this.createKeywordVector(conv.userMessage + '\n' + conv.aiResponse);
                        similarity = this.calculateCosineSimilarity(currentVector, convVector);
                    } else if (!isConvArray && isCurrArray) {
                        // 库中是对象，当前是数组 -> 将库中向量转为数组（暂不支持，返回低相似度）
                        console.warn(`[向量检索] 第${conv.turnIndex}轮向量类型不兼容，跳过`);
                        similarity = 0;
                    }
                }
                
                return {
                    index: index,
                    turnIndex: conv.turnIndex,
                    similarity: similarity,
                    conversation: conv,
                    vectorType: isConvArray ? 'dense' : 'sparse'
                };
            });
            
            // 4. 过滤并排序
            const relevantConversations = similarities
                .filter(item => item.similarity >= this.minSimilarityThreshold)
                .sort((a, b) => b.similarity - a.similarity)
                .slice(0, this.maxRetrieveCount);
            
            // 🆕 统计向量类型信息
            const denseCount = similarities.filter(s => s.vectorType === 'dense').length;
            const sparseCount = similarities.filter(s => s.vectorType === 'sparse').length;
            
            console.log(`╔════════════════════════════════════════════════╗`);
            console.log(`║  🔍 向量检索执行报告                            ║`);
            console.log(`╠════════════════════════════════════════════════╣`);
            console.log(`║  📊 向量库统计：                                ║`);
            console.log(`║    - 总记录数：${this.conversationEmbeddings.length}轮                          ║`);
            console.log(`║    - 稠密向量（Dense）：${denseCount}轮                    ║`);
            console.log(`║    - 稀疏向量（Sparse）：${sparseCount}轮                   ║`);
            console.log(`║  🎯 检索结果：                                  ║`);
            console.log(`║    - 检索方法：${vectorLibType === 'dense' ? '关键词降级' : '关键词匹配'}         ║`);
            console.log(`║    - 找到记录：${relevantConversations.length}条                            ║`);
            console.log(`╚════════════════════════════════════════════════╝`);
            
            relevantConversations.forEach(item => {
                const typeTag = item.vectorType === 'dense' ? '[稠密→转换]' : '[稀疏]';
                console.log(`  ${typeTag} 第${item.turnIndex}轮 相似度:${item.similarity.toFixed(3)} ${item.conversation.summary}`);
            });
            
            // 4. 格式化为上下文
            const relevantChunks = relevantConversations.map(item => ({
                turnIndex: item.turnIndex,
                userMessage: item.conversation.userMessage,
                aiResponse: item.conversation.aiResponse,
                similarity: item.similarity,
                summary: item.conversation.summary
            }));
            
            return {
                relevantChunks: relevantChunks,
                recentChunks: recentHistory
            };
            
        } catch (error) {
            console.error('[向量检索] 检索失败:', error);
            return {
                relevantChunks: [],
                recentChunks: recentHistory
            };
        }
    }

    /**
     * 构建优化后的上下文消息
     */
    async buildOptimizedMessages(systemPrompt, currentVariables, currentInput, historyDepth = 3, fullConversationHistory = []) {
        const messages = [];
        
        // 1. 系统提示词
        messages.push({
            role: 'system',
            content: systemPrompt
        });
        
        // 2. 当前变量状态
        messages.push({
            role: 'system',
            content: '当前角色变量状态：\n```json\n' + JSON.stringify(currentVariables, null, 2) + '\n```'
        });
        
        // 3. 检索相关历史（远期记忆）
        const retrievalResult = await this.retrieveRelevantContext(currentInput, []);
        
        if (retrievalResult.relevantChunks.length > 0) {
            let relevantContext = '【相关历史回忆】以下是与当前情境相关的过往记忆：\n\n';
            
            retrievalResult.relevantChunks.forEach((chunk, index) => {
                relevantContext += `记忆${index + 1}（第${chunk.turnIndex}轮对话，相似度${(chunk.similarity * 100).toFixed(1)}%）：\n`;
                relevantContext += `- 玩家行动：${chunk.userMessage}\n`;
                relevantContext += `- 剧情摘要：${chunk.summary}\n\n`;
            });
            
            messages.push({
                role: 'system',
                content: relevantContext
            });
        }
        
        // 4. 最近对话（近期记忆）- 使用传入的完整历史记录
        const conversationHistory = fullConversationHistory.length > 0 
            ? fullConversationHistory 
            : (window.gameState?.conversationHistory || []);
            
        if (conversationHistory.length > 0 && historyDepth > 0) {
            const recentHistory = conversationHistory.slice(-historyDepth * 2);
            messages.push(...recentHistory);
            console.log(`[近期记忆] 添加最近${recentHistory.length}条对话（历史层数设置：${historyDepth}）`);
        }
        
        // 5. 当前用户输入
        messages.push({
            role: 'user',
            content: currentInput
        });
        
        const totalHistory = conversationHistory.length;
        const totalTurns = Math.floor(totalHistory / 2);
        const recentCount = Math.min(historyDepth * 2, conversationHistory.length);
        const recentTurns = Math.floor(recentCount / 2);
        const vectorCount = retrievalResult.relevantChunks.length;
        
        console.log(`╔════════════════════════════════════════════════╗`);
        console.log(`║  🧬 向量检索上下文构建报告                      ║`);
        console.log(`╠════════════════════════════════════════════════╣`);
        console.log(`║  📊 原始数据：                                  ║`);
        console.log(`║    - 总对话轮数：${totalTurns}轮（${totalHistory}条消息）`);
        console.log(`║    - 历史层数设置：${historyDepth}层              ║`);
        console.log(`║    - 向量库大小：${this.conversationEmbeddings.length}轮   ║`);
        console.log(`╠════════════════════════════════════════════════╣`);
        console.log(`║  📤 实际发送：                                  ║`);
        console.log(`║    ✓ 系统消息：2条（提示词+变量）              ║`);
        console.log(`║    ✓ 向量检索：${vectorCount}条（远期相关记忆）           ║`);
        console.log(`║    ✓ 最近对话：${recentTurns}轮=${recentCount}条（近期连贯记忆）    ║`);
        console.log(`║    ✓ 当前输入：1条                             ║`);
        console.log(`╠════════════════════════════════════════════════╣`);
        console.log(`║  💡 总消息数：${messages.length}条                          ║`);
        console.log(`║  💰 Token节省：约${totalHistory > 0 ? Math.round((1 - recentCount/totalHistory) * 100) : 0}%                    ║`);
        console.log(`╚════════════════════════════════════════════════╝`);
        
        return messages;
    }

    /**
     * 【方案2】通过API获取embedding（需要配置额外API）
     */
    async getEmbeddingFromAPI(text) {
        // 检查是否启用了额外API
        if (!window.extraApiConfig || !window.extraApiConfig.enabled) {
            console.warn('[向量API] 额外API未启用，回退到关键词方法');
            return this.createKeywordVector(text);
        }
        
        try {
            const endpoint = window.extraApiConfig.endpoint.trim().replace(/\/+$/, '');
            const apiKey = window.extraApiConfig.key;
            
            // OpenAI embeddings API
            const response = await fetch(`${endpoint}/embeddings`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${apiKey}`
                },
                body: JSON.stringify({
                    input: text.substring(0, 8000), // 限制长度
                    model: 'text-embedding-ada-002' // 可配置
                })
            });
            
            if (!response.ok) {
                throw new Error(`API错误: ${response.status}`);
            }
            
            const data = await response.json();
            return data.data[0].embedding; // 返回向量数组
            
        } catch (error) {
            console.error('[向量API] 调用失败:', error);
            // 回退到关键词方法
            return this.createKeywordVector(text);
        }
    }

    /**
     * 【方案3】使用transformers.js（浏览器端模型）
     * 需要先加载：window.loadTransformersJS()
     */
    async getEmbeddingFromTransformers(text) {
        try {
            // 检查库是否加载
            if (typeof window.transformers === 'undefined' && typeof window.loadTransformersJS === 'function') {
                console.log('[Transformers.js] 正在加载库（首次加载）...');
                await window.loadTransformersJS();
            }
            
            if (typeof window.transformers === 'undefined') {
                console.warn('[Transformers.js] 库加载失败，回退到关键词方法');
                return this.createKeywordVector(text);
            }
            
            // 使用轻量级多语言模型
            const { pipeline } = window.transformers;
            
            if (!this.embeddingPipeline) {
                console.log('[Transformers.js] 正在加载模型（首次约50MB，请耐心等待）...');
                
                // 显示加载提示
                if (typeof window !== 'undefined' && window.document) {
                    const loadingMsg = document.createElement('div');
                    loadingMsg.id = 'transformersLoading';
                    loadingMsg.style.cssText = `
                        position: fixed;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        background: white;
                        padding: 30px;
                        border-radius: 15px;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                        z-index: 10001;
                        text-align: center;
                    `;
                    loadingMsg.innerHTML = `
                        <div style="color: #667eea; font-size: 20px; font-weight: bold; margin-bottom: 15px;">
                            🤖 正在加载AI模型...
                        </div>
                        <div style="color: #666; font-size: 14px;">
                            首次加载约50MB，请耐心等待<br>
                            模型会缓存到浏览器，下次秒开
                        </div>
                        <div class="loading" style="margin: 20px auto;"></div>
                    `;
                    document.body.appendChild(loadingMsg);
                }
                
                this.embeddingPipeline = await pipeline(
                    'feature-extraction', 
                    'Xenova/paraphrase-multilingual-MiniLM-L12-v2'
                );
                
                // 移除加载提示
                const loadingMsg = document.getElementById('transformersLoading');
                if (loadingMsg) loadingMsg.remove();
                
                console.log('[Transformers.js] ✅ 模型加载完成！');
            }
            
            // 生成向量
            const output = await this.embeddingPipeline(text.substring(0, 500), {
                pooling: 'mean',
                normalize: true
            });
            
            // 转换为普通数组
            const vector = Array.from(output.data);
            
            console.log(`[Transformers.js] 向量生成成功（维度：${vector.length}）`);
            
            return vector;
            
        } catch (error) {
            console.error('[Transformers.js] 错误:', error);
            
            // 移除加载提示（如果存在）
            const loadingMsg = document.getElementById('transformersLoading');
            if (loadingMsg) loadingMsg.remove();
            
            // 回退到关键词方法
            console.warn('[Transformers.js] 回退到关键词方法');
            return this.createKeywordVector(text);
        }
    }

    /**
     * 切换embedding方法
     */
    setEmbeddingMethod(method) {
        if (['keyword', 'api', 'transformers'].includes(method)) {
            this.embeddingMethod = method;
            console.log(`[向量方法] 已切换到: ${method}`);
        } else {
            console.error('[向量方法] 无效的方法:', method);
        }
    }

    /**
     * 清空向量库
     */
    clear() {
        this.conversationEmbeddings = [];
        console.log('[向量库] 已清空');
    }

    /**
     * 保存向量库到IndexedDB
     */
    async saveToIndexedDB(dbName = 'xiuxian_vector_db') {
        try {
            const db = await this.openVectorDB(dbName);
            const transaction = db.transaction(['embeddings'], 'readwrite');
            const store = transaction.objectStore('embeddings');
            
            await store.clear();
            await store.put({
                id: 'main',
                embeddings: this.conversationEmbeddings,
                timestamp: Date.now()
            });
            
            console.log('[向量库] 已保存到IndexedDB');
        } catch (error) {
            console.error('[向量库] 保存失败:', error);
        }
    }

    /**
     * 从IndexedDB加载向量库
     */
    async loadFromIndexedDB(dbName = 'xiuxian_vector_db') {
        try {
            const db = await this.openVectorDB(dbName);
            const transaction = db.transaction(['embeddings'], 'readonly');
            const store = transaction.objectStore('embeddings');
            
            const request = store.get('main');
            const result = await new Promise((resolve, reject) => {
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
            
            if (result && result.embeddings) {
                this.conversationEmbeddings = result.embeddings;
                console.log(`[向量库] 已从IndexedDB加载${this.conversationEmbeddings.length}条记录`);
            }
        } catch (error) {
            console.error('[向量库] 加载失败:', error);
        }
    }

    /**
     * 打开向量数据库
     */
    openVectorDB(dbName) {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(dbName, 1);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('embeddings')) {
                    db.createObjectStore('embeddings', { keyPath: 'id' });
                }
            };
        });
    }
}

// 创建全局实例
window.contextVectorManager = new ContextVectorManager();

console.log('[向量系统] 已加载，使用方法：');
console.log('1. window.contextVectorManager.setEmbeddingMethod("keyword") - 设置向量化方法');
console.log('2. 在每次AI对话后自动调用 addConversation() 添加到向量库');
console.log('3. 在构建上下文时调用 buildOptimizedMessages() 获取优化后的上下文');

