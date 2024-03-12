/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

using UnityEngine;
using System.Collections;

namespace Facebook.WitAi.Utilities
{
    public static class CoroutineUtility
    {
        // Start coroutine
        public static CoroutinePerformer StartCoroutine(IEnumerator asyncMethod)
        {
            CoroutinePerformer performer = GetPerformer();
            performer.CoroutineBegin(asyncMethod);
            return performer;
        }
        // Get performer
        private static CoroutinePerformer GetPerformer()
        {
            CoroutinePerformer performer = new GameObject("Coroutine").AddComponent<CoroutinePerformer>();
            //performer.gameObject.hideFlags = HideFlags.DontSave;
            performer.gameObject.hideFlags = HideFlags.HideAndDontSave;
            return performer;
        }
        // Coroutine performer
        public class CoroutinePerformer : MonoBehaviour
        {
            // Coroutine
            public bool IsRunning { get; private set; }
            private Coroutine _runtimeCoroutine;

            // Dont destroy
            private void Awake()
            {
                DontDestroyOnLoad(gameObject);
            }

            // Perform coroutine
            public void CoroutineBegin(IEnumerator asyncMethod)
            {
                // Cannot call twice
                if (IsRunning)
                {
                    return;
                }

                // Begin running
                IsRunning = true;

#if UNITY_EDITOR
                // Editor mode
                if (!Application.isPlaying)
                {
                    _editorMethod = asyncMethod;
                    UnityEditor.EditorApplication.update += EditorCoroutineIterate;
                    return;
                }
#endif

                // Begin coroutine
                _runtimeCoroutine = StartCoroutine(RuntimeCoroutineIterate(asyncMethod));
            }

#if UNITY_EDITOR
            // Editor iterate
            private IEnumerator _editorMethod;
            private void EditorCoroutineIterate()
            {
                if (_editorMethod != null)
                {
                    if (!_editorMethod.MoveNext())
                    {
                        CoroutineComplete();
                    }
                }
            }
#endif
            // Runtime iterate
            private IEnumerator RuntimeCoroutineIterate(IEnumerator asyncMethod)
            {
                // Wait for completion
                yield return StartCoroutine(asyncMethod);
                // Complete
                CoroutineComplete();
            }

            // Cancel on destroy
            private void OnDestroy()
            {
                if (IsRunning)
                {
                    CoroutineCancel();
                }
            }
            // Cancel coroutine
            public void CoroutineCancel()
            {
                if (IsRunning)
                {
                    CoroutineComplete();
                }
            }
            // Completed
            private void CoroutineComplete()
            {
                // Ignore unless running
                if (!IsRunning)
                {
                    return;
                }

                // Done
                IsRunning = false;

#if UNITY_EDITOR
                // Complete
                if (_editorMethod != null)
                {
                    UnityEditor.EditorApplication.update -= EditorCoroutineIterate;
                    _editorMethod = null;
                }
#endif

                // Stop coroutine
                if (_runtimeCoroutine != null)
                {
                    StopCoroutine(_runtimeCoroutine);
                    _runtimeCoroutine = null;
                }

                // Destroy
                DestroyImmediate(gameObject);
            }
        }
    }
}
