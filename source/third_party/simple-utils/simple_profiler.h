//////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                              //
// Copyright (c) 2019 Tencent.com, Inc. All Rights Reserved                                     //
// Author: Aster Jian (asterjian@qq.com)                                                        //
//                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "simple_table.h"

namespace utils
{
    struct Scope;

    /**
     * \brief 
     */
    class Profiler
    {
    public:
        /**
         * \brief 构造函数
         */
        Profiler(const char* name);

        /**
         * \brief 析构函数
         */
        ~Profiler();

        /**
         * \brief 打印 Profile 的结果
         */
        void Print(std::ostream& os = std::cout);

    private:
        /**
         * \brief 进入Scope
         */
        void EnterScope(const char* name);

        /**
         * \brief 退出 Scope
         */
        void LeaveScope();

        friend struct ScopeHolder;

        /**
         * \brief 
         */
        static thread_local Profiler* s_instance;

        const char* name_;

        /**
         * \brief 跟节点
         */
        Scope* root_{ nullptr };

        /**
         * \brief 当前节点
         */
        Scope* current_{ nullptr };
    };

    /**
     * \brief Scope Holder
     */
    struct ScopeHolder
    {
        ScopeHolder(const char* scope_name)
        {
            // 自动调用进入 Scope 函数
            if (Profiler::s_instance)
            {
                Profiler::s_instance->EnterScope(scope_name);
            }
        }

        ~ScopeHolder()
        {
            // 自动调用退出 Scope 函数
            if (Profiler::s_instance)
            {
                Profiler::s_instance->LeaveScope();
            }
        }
    };
}

#if TRT_INFER_ENABLE_PROFILING
#define UTILS_PROFILE(name) utils::ScopeHolder scope_holder_##name(#name)
#else
#define UTILS_PROFILE(name)
#endif // TRT_INFER_ENABLE_PROFILING


